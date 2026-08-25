// Copyright (c) 2023 - 2026 Restate Software, Inc., Restate GmbH.
// All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! AWS-to-GCP workload identity federation for minting Google ID tokens without storing Google
//! credentials in Restate.
//!
//! The trust chain is:
//!
//! ```text
//! ambient AWS credentials
//!   -> sts:AssumeRole(operator-configured AWS federation role)
//!   -> SigV4-signed GetCallerIdentity envelope (AIP-4117 aws4_request)
//!   -> Google STS token exchange at the deployment's workload identity provider
//!   -> IAM Credentials generateIdToken as the deployment's service account
//! ```
//!
//! The assumed AWS role session is shared across the process. Everything after it is scoped by a
//! deployment's `workload_identity_provider` and `impersonate_service_account`.

use std::fmt;
use std::sync::{Arc, Weak};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_config::sts::AssumeRoleProvider;
use aws_credential_types::Credentials as AwsCredentials;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use google_cloud_auth::credentials::external_account::ProgrammaticBuilder;
use google_cloud_auth::credentials::idtoken;
use google_cloud_auth::credentials::subject_token::{
    Builder as SubjectTokenBuilder, SubjectToken, SubjectTokenProvider,
};
use google_cloud_auth::errors::SubjectTokenProviderError;
use metrics::counter;
use tokio::sync::{Mutex, OnceCell};
use tracing::warn;

use restate_types::config::GcpFederationOptions;

use crate::metric_definitions::{GCP_FEDERATION_SUBJECT_TOKENS, RESULT_ERROR, RESULT_SUCCESS};

use super::{GcpAuthError, IdTokenSource, IdTokenSpec, RecoverableCell};

/// AWS subject-token type Google STS expects for a SigV4-signed `GetCallerIdentity` envelope.
const AWS4_SUBJECT_TOKEN_TYPE: &str = "urn:ietf:params:aws:token-type:aws4_request";
const GOOGLE_STS_TOKEN_URL: &str = "https://sts.googleapis.com/v1/token";

/// Refresh the assumed AWS role session before it expires so subject-token generation does not
/// race its expiry.
const AWS_ROLE_REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// The process-wide `[gcp-federation]` config, installed once from `ServiceClient` construction
/// (see [`install_config`]). Unset means the operator never configured the block: every federated
/// construction then fails with a permanent, actionable [`GcpAuthError::Build`] rather than
/// falling back to an unauthenticated request.
///
/// Operator-owned; see [`GcpFederationOptions`] for the security rationale for why this
/// configuration can only ever come from the operator, never from a deployment registration.
///
/// This stays process-wide because it is operator configuration, not state owned by a TaskCenter.
/// It is not live-reloadable: already-constructed credentials retain the AWS federation identity
/// used at construction, so changes are logged and ignored until restart.
static FEDERATION_CONFIG: std::sync::OnceLock<GcpFederationOptions> = std::sync::OnceLock::new();

/// Result of comparing incoming configuration with the process-wide value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigInstallOutcome {
    NotRequested,
    FirstInstall,
    Unchanged,
    Differing,
    RemovalIgnored,
}

/// Decide whether to install before validating. Reloaded values are ignored, so rejecting an
/// invalid replacement would fail service-client construction without changing the active value.
fn decide_config_install(
    installed: Option<&GcpFederationOptions>,
    incoming: Option<&GcpFederationOptions>,
) -> ConfigInstallOutcome {
    match (installed, incoming) {
        (None, None) => ConfigInstallOutcome::NotRequested,
        (None, Some(_)) => ConfigInstallOutcome::FirstInstall,
        (Some(installed), Some(incoming)) if installed == incoming => {
            ConfigInstallOutcome::Unchanged
        }
        (Some(_), Some(_)) => ConfigInstallOutcome::Differing,
        (Some(_), None) => ConfigInstallOutcome::RemovalIgnored,
    }
}

/// Describe only the fields that differ for the reload warning.
fn describe_config_diff(
    installed: &GcpFederationOptions,
    incoming: &GcpFederationOptions,
) -> String {
    let mut changes = Vec::new();
    if installed.aws_role_arn != incoming.aws_role_arn {
        changes.push(format!(
            "aws-role-arn '{}' -> '{}'",
            installed.aws_role_arn, incoming.aws_role_arn
        ));
    }
    if installed.aws_role_session_name != incoming.aws_role_session_name {
        changes.push(format!(
            "aws-role-session-name '{}' -> '{}'",
            installed.aws_role_session_name, incoming.aws_role_session_name
        ));
    }
    changes.join(", ")
}

/// Install and validate the process-wide configuration. Later changes are logged and ignored
/// until restart; validating ignored replacements would unnecessarily fail the caller.
pub(crate) fn install_config(config: Option<GcpFederationOptions>) -> Result<(), String> {
    match decide_config_install(FEDERATION_CONFIG.get(), config.as_ref()) {
        ConfigInstallOutcome::NotRequested | ConfigInstallOutcome::Unchanged => {}
        ConfigInstallOutcome::FirstInstall => {
            let config = config.expect("FirstInstall implies a config was given");
            validate_aws_role_arn(&config.aws_role_arn)?;
            validate_aws_role_session_name(&config.aws_role_session_name)?;
            let _ = FEDERATION_CONFIG.set(config);
        }
        ConfigInstallOutcome::Differing => {
            let installed = FEDERATION_CONFIG
                .get()
                .expect("Differing implies a config is already installed");
            let incoming = config.expect("Differing implies a config was given");
            tracing::warn!(
                "[gcp-federation] configuration changed ({}) but this block is not \
                 live-reloadable; the configuration active since process start (aws-role-arn \
                 '{}') remains in use until the server is restarted",
                describe_config_diff(installed, &incoming),
                installed.aws_role_arn,
            );
        }
        ConfigInstallOutcome::RemovalIgnored => {
            let installed = FEDERATION_CONFIG
                .get()
                .expect("RemovalIgnored implies a config is already installed");
            tracing::warn!(
                "[gcp-federation] configuration block was removed, but this block is not \
                 live-reloadable; the configuration active since process start (aws-role-arn \
                 '{}') remains in use until the server is restarted",
                installed.aws_role_arn,
            );
        }
    }
    Ok(())
}

/// Validate the expected IAM role ARN shape so configuration errors fail at startup rather than
/// at the first `AssumeRole` request. The AWS SDK does not expose a public ARN parser.
fn validate_aws_role_arn(arn: &str) -> Result<(), String> {
    let invalid = || {
        format!(
            "aws-role-arn '{arn}' is not a valid AWS IAM role ARN; expected \
             arn:aws:iam::<12-digit account id>:role/<role-name-or-path>"
        )
    };
    let parts: Vec<&str> = arn.split(':').collect();
    let [scheme, partition, service, region, account, resource] = parts.as_slice() else {
        return Err(invalid());
    };
    let partition_suffix_ok = partition.strip_prefix("aws").is_some_and(|suffix| {
        suffix
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    });
    if *scheme != "arn"
        || !partition_suffix_ok
        || *service != "iam"
        || !region.is_empty()
        || account.len() != 12
        || !account.bytes().all(|b| b.is_ascii_digit())
        || !resource.starts_with("role/")
        || resource.len() <= "role/".len()
    {
        return Err(invalid());
    }
    Ok(())
}

/// Validate AWS STS `RoleSessionName` constraints at startup. The generated SDK builder leaves
/// these service-side constraints unchecked.
fn validate_aws_role_session_name(name: &str) -> Result<(), String> {
    let len = name.chars().count();
    let chars_ok = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "_+=,.@-".contains(c));
    if !(2..=64).contains(&len) || !chars_ok {
        return Err(format!(
            "aws-role-session-name '{name}' is not a valid AWS STS RoleSessionName; expected \
             2-64 characters from [A-Za-z0-9_+=,.@-]"
        ));
    }
    Ok(())
}

/// Cached credentials for the process-wide AWS federation role session, reused by every
/// [`AwsSubjectTokenProvider`] and constructed on first use.
/// `provider` is type-erased behind `SharedCredentialsProvider` (rather than the concrete
/// `AssumeRoleProvider`) so tests can wrap a fixed [`AwsCredentials`] value directly instead of
/// driving the real AWS SDK config/STS machinery.
struct AwsFederationCredentials {
    /// Resolved once from the AWS SDK default chain.
    region: String,
    provider: SharedCredentialsProvider,
    cached: Mutex<Option<AwsCredentials>>,
}

impl fmt::Debug for AwsFederationCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AwsFederationCredentials")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

impl AwsFederationCredentials {
    async fn init(config: &GcpFederationOptions) -> Result<Self, String> {
        let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let region = sdk_config
            .region()
            .map(|region| region.to_string())
            .ok_or_else(|| {
                "no AWS region resolved from the SDK default chain; the region is required to \
                 sign the GetCallerIdentity subject token"
                    .to_owned()
            })?;
        let provider = AssumeRoleProvider::builder(config.aws_role_arn.clone())
            .configure(&sdk_config)
            .session_name(config.aws_role_session_name.clone())
            .build()
            .await;
        Ok(Self {
            region,
            provider: SharedCredentialsProvider::new(provider),
            cached: Mutex::new(None),
        })
    }

    /// Returns the current AWS role session credentials, refreshing them via `sts:AssumeRole` if
    /// the cached session is absent or within [`AWS_ROLE_REFRESH_MARGIN`] of expiry. Shared across
    /// every federated deployment's `subject_token()` calls, so a fleet of federated deployments
    /// refreshing around the same time coalesces into the one `AssumeRole` call each needs rather
    /// than one per deployment.
    async fn credentials(&self) -> Result<AwsCredentials, FederationError> {
        let mut guard = self.cached.lock().await;
        if let Some(creds) = guard.as_ref() {
            let fresh_enough = creds
                .expiry()
                .is_none_or(|expiry| expiry > SystemTime::now() + AWS_ROLE_REFRESH_MARGIN);
            if fresh_enough {
                return Ok(creds.clone());
            }
        }
        let fresh = self
            .provider
            .provide_credentials()
            .await
            .map_err(|e| federation_error_from_assume_role_failure(&e))?;
        *guard = Some(fresh.clone());
        Ok(fresh)
    }
}

/// Classifies an `sts:AssumeRole` failure surfaced by `AssumeRoleProvider::provide_credentials()`.
/// Authorization failures (trust policy or IAM denying `sts:AssumeRole`) are permanent: no amount
/// of retrying fixes them without an operator changing IAM, and classifying them as transient
/// would have the refresh loop retry an `AccessDenied` forever instead of publishing a permanent
/// error and exiting -- which is what lets [`RecoverableCell`]'s probe-and-replace recovery
/// rebuild the source once IAM is fixed. Everything else (network/dispatch/timeout/chain
/// resolution) genuinely can resolve on retry and stays transient.
fn federation_error_from_assume_role_failure(
    error: &aws_credential_types::provider::error::CredentialsError,
) -> FederationError {
    let message = format!("assuming the AWS federation role for GCP authentication: {error}");
    let access_denied = matches!(
        assume_role_error_code(error),
        Some("AccessDenied" | "AccessDeniedException")
    );
    if access_denied {
        FederationError::permanent(message)
    } else {
        FederationError::transient(message)
    }
}

/// Walks `error`'s source chain for the `SdkError<AssumeRoleError>` `AssumeRoleProvider` wraps its
/// STS response errors in, and returns the STS wire error code from it, if any. `AssumeRole`'s
/// `AccessDenied` has no modeled exception variant in `AssumeRoleError`, so it always surfaces
/// through the catch-all `Unhandled` variant; `ProvideErrorMetadata::code()` still recovers the
/// wire error code for unmodeled errors, which is why this classifies on that structured field
/// rather than matching the human-readable `Display` text or the service-error enum's variants.
fn assume_role_error_code<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a str> {
    use aws_sdk_sts::error::{ProvideErrorMetadata, SdkError};
    use aws_sdk_sts::operation::assume_role::AssumeRoleError;

    let mut cause = Some(error);
    while let Some(err) = cause {
        if let Some(sdk_error) = err.downcast_ref::<SdkError<AssumeRoleError>>() {
            return sdk_error.code();
        }
        cause = err.source();
    }
    None
}

static AWS_FEDERATION_CREDENTIALS: OnceCell<Arc<AwsFederationCredentials>> = OnceCell::const_new();

/// Returns the shared AWS federation credentials, constructing them on first use. Construction failure (missing
/// `[gcp-federation]` config, or no AWS region resolvable) is not cached:
/// [`OnceCell::get_or_try_init`] leaves the cell empty on `Err`, so the next attempt retries.
///
/// Stays a plain process-wide static for the same reason [`FEDERATION_CONFIG`] does: it holds AWS
/// credential state (an `AssumeRoleProvider` plus a cached session) with no background refresh
/// task of its own. Nothing in [`AwsFederationCredentials::init`] is tied to a TaskCenter runtime,
/// so TaskCenter replacement cannot leave stale refresh work behind.
async fn aws_federation_credentials() -> Result<Arc<AwsFederationCredentials>, String> {
    AWS_FEDERATION_CREDENTIALS
        .get_or_try_init(|| async {
            let Some(config) = FEDERATION_CONFIG.get() else {
                return Err(
                    "this deployment requests GCP workload identity federation, but the server \
                     has no [gcp-federation] configuration; set aws-role-arn and \
                     aws-role-session-name to enable it"
                        .to_owned(),
                );
            };
            AwsFederationCredentials::init(config).await.map(Arc::new)
        })
        .await
        .cloned()
}

/// Error from a step of the federation chain, carrying the transient/permanent classification
/// [`google_cloud_auth`]'s external-account refresh loop uses to decide whether to keep retrying.
#[derive(Debug)]
struct FederationError {
    transient: bool,
    message: String,
}

impl FederationError {
    fn transient(message: String) -> Self {
        Self {
            transient: true,
            message,
        }
    }

    fn permanent(message: String) -> Self {
        Self {
            transient: false,
            message,
        }
    }
}

impl fmt::Display for FederationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FederationError {}

impl SubjectTokenProviderError for FederationError {
    fn is_transient(&self) -> bool {
        self.transient
    }
}

/// Supplies the AIP-4117 AWS subject token to `google-cloud-auth`'s external-account credential
/// on each refresh. It only retains the shared [`AwsFederationCredentials`]; it does not read or
/// write `AWS_*` environment variables or modify process-global AWS SDK state.
#[derive(Debug)]
struct AwsSubjectTokenProvider {
    aws_federation_credentials: Arc<AwsFederationCredentials>,
    /// The full resource name of the customer's workload identity provider. Doubles as the
    /// `x-goog-cloud-target-resource` header value that binds the signed envelope to this pool
    /// (see `build_subject_token`) and as the STS `audience` parameter.
    provider_resource: String,
}

impl SubjectTokenProvider for AwsSubjectTokenProvider {
    type Error = FederationError;

    async fn subject_token(&self) -> Result<SubjectToken, Self::Error> {
        let result = self.subject_token_inner().await;
        counter!(
            GCP_FEDERATION_SUBJECT_TOKENS,
            "result" => if result.is_ok() { RESULT_SUCCESS } else { RESULT_ERROR }
        )
        .increment(1);
        result
    }
}

impl AwsSubjectTokenProvider {
    async fn subject_token_inner(&self) -> Result<SubjectToken, FederationError> {
        let credentials = self.aws_federation_credentials.credentials().await?;
        let envelope = build_subject_token(
            &credentials,
            &self.aws_federation_credentials.region,
            &self.provider_resource,
            SystemTime::now(),
        )
        .map_err(|e| {
            FederationError::permanent(format!("signing the GetCallerIdentity subject token: {e}"))
        })?;

        // google-cloud-auth's built-in AWS credential source (`external_account_sources::aws_sourced`)
        // form-urlencodes the AIP-4117 JSON envelope before returning it as the subject token; the
        // STS token-exchange request then form-encodes the whole request body for transport, so the
        // envelope arrives at Google encoded exactly twice: once by us, once by the transport layer.
        // We match that convention here so the two encoding layers are exactly as Google expects,
        // pinned by `federation_tests::subject_token_matches_aws_sourced_encoding_convention`.
        let subject_token: String =
            url::form_urlencoded::byte_serialize(envelope.as_bytes()).collect();
        Ok(SubjectTokenBuilder::new(subject_token).build())
    }
}

/// Sign a `GetCallerIdentity` call with `credentials` and render the AIP-4117 subject-token
/// envelope. `target_resource` (the workload identity provider resource name) travels as the
/// `x-goog-cloud-target-resource` header and is bound inside `SignedHeaders`: that binding is
/// what stops a signed envelope minted for one workload identity pool from being replayed against
/// another.
fn build_subject_token(
    credentials: &AwsCredentials,
    region: &str,
    target_resource: &str,
    signing_time: SystemTime,
) -> Result<String, String> {
    let url =
        format!("https://sts.{region}.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15");
    let host = format!("sts.{region}.amazonaws.com");

    let identity = credentials.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name("sts")
        .time(signing_time)
        .settings(SigningSettings::default())
        .build()
        .map_err(|e| format!("building sigv4 signing params: {e}"))?;

    // Headers presented for signing; the signer canonicalises, so order does not matter here.
    let headers = vec![
        ("host", host.as_str()),
        ("x-goog-cloud-target-resource", target_resource),
    ];

    let signable =
        SignableRequest::new("POST", &url, headers.into_iter(), SignableBody::Bytes(b""))
            .map_err(|e| format!("building signable request: {e}"))?;

    let (instructions, _signature) = sign(signable, &params.into())
        .map_err(|e| format!("signing GetCallerIdentity: {e}"))?
        .into_parts();

    let mut out = vec![
        SubjectTokenHeader {
            key: "host".to_owned(),
            value: host,
        },
        SubjectTokenHeader {
            key: "x-goog-cloud-target-resource".to_owned(),
            value: target_resource.to_owned(),
        },
    ];
    for (name, value) in instructions.headers() {
        out.push(SubjectTokenHeader {
            key: name.to_owned(),
            value: value.to_owned(),
        });
    }

    let envelope = SubjectTokenEnvelope {
        url,
        method: "POST",
        headers: out,
        body: String::new(),
    };
    serde_json::to_string(&envelope).map_err(|e| format!("serializing subject token: {e}"))
}

/// The AIP-4117 envelope Google STS expects for `aws4_request` subject tokens: a JSON description
/// of a signed `GetCallerIdentity` request.
#[derive(Debug, serde::Serialize)]
struct SubjectTokenEnvelope {
    url: String,
    method: &'static str,
    headers: Vec<SubjectTokenHeader>,
    body: String,
}

#[derive(Debug, serde::Serialize)]
struct SubjectTokenHeader {
    key: String,
    value: String,
}

/// The shared Google STS access-token source for one WIF provider: the external-account
/// credential produced by exchanging a SigV4-signed AWS subject token, before impersonation. Its
/// lifetime is not owned by this struct -- see [`FederatedAccessTokenSources`] for the ownership
/// model.
///
/// Deliberately holds no reference back to `CredentialRegistry` or any task-center-bound state:
/// `ClearRegistrySlotOnDrop`'s `Weak`-based shutdown guard must stay cycle-free.
pub(super) struct FederatedAccessTokenSource {
    pub(super) credentials: RecoverableCell<google_cloud_auth::credentials::Credentials>,
}

/// The shared, weak-indexed home for every WIF provider's [`FederatedAccessTokenSource`], keyed by
/// provider resource name.
///
/// Lifetime is entirely reference-driven: this map holds only `Weak` entries, and every
/// [`FederatedIdTokenCredentials`] built for a provider holds a strong `Arc` to that provider's
/// source for as long as it itself stays cached (its `_access_token_source` field) --
/// [`federated_access_token_source`] upgrades a live entry or replaces a dead/absent one, and
/// [`reap_disused_federated_access_token_sources`] prunes whatever no longer upgrades. Time-based
/// eviction can't substitute for this: this map is touched only from outer construction, never
/// from a steady-state mint against an already-built outer credential, so an idle timer would have
/// nothing to observe -- an outer credential's own cloned copy of its access-token `Credentials`
/// is invisible to this map.
///
/// The provider resource name is a sufficient key while the process has exactly one configured
/// AWS federation identity. If deployments can later select among AWS identities, that identity
/// must become part of the key: the same provider reached through a different AWS identity is a
/// distinct access-token source.
pub(super) type FederatedAccessTokenSources =
    parking_lot::Mutex<std::collections::HashMap<String, Weak<FederatedAccessTokenSource>>>;

/// The outer, per-audience/service-account federated ID-token credential. Holds a required lease
/// on its provider's [`FederatedAccessTokenSource`] (see that type's doc for the ownership model)
/// rather than an optional field on `Live`, so the federated construction path cannot compile
/// without keeping it.
pub(super) struct FederatedIdTokenCredentials {
    credentials: google_cloud_auth::credentials::idtoken::IDTokenCredentials,
    // See FederatedAccessTokenSources's doc: dropping this lease (when the outer moka cache
    // evicts the entry this credential lives in) is what housekeeping's reap observes.
    _access_token_source: Arc<FederatedAccessTokenSource>,
}

#[async_trait]
impl IdTokenSource for FederatedIdTokenCredentials {
    async fn id_token(&self) -> Result<String, google_cloud_auth::errors::CredentialsError> {
        self.credentials.id_token().await
    }
}

/// Assembles the federation chain for `spec` (whose `wif_provider` is `Some`) and returns the
/// resulting [`FederatedIdTokenCredentials`]: (1) resolve this provider's shared
/// [`FederatedAccessTokenSource`] from `sources`; (2) resolve or build its access-token
/// credentials through that source's [`RecoverableCell`] (single-flighted, so N concurrent cold
/// keys for one provider share one build); (3) build the impersonated ID-token credential from the
/// *cloned* access-token credentials; (4) return both together. Never clones the access-token
/// credentials out and discards the `Arc<FederatedAccessTokenSource>` from step 1 -- holding that
/// lease is the entire point.
pub(super) async fn build_federated_source(
    sources: &FederatedAccessTokenSources,
    spec: IdTokenSpec,
) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
    let IdTokenSpec {
        wif_provider,
        impersonate,
        audience,
    } = spec;
    let wif_provider = wif_provider.expect("build_federated_source called with wif_provider unset");
    let Some(impersonate) = impersonate else {
        // The schema registry requires `impersonate_service_account` whenever
        // `workload_identity_provider` is set (the external-account credential this chain
        // produces cannot mint an ID token ambiently), so this is unreachable via registration —
        // guarded here defensively for callers that bypass the registry.
        return Err(GcpAuthError::Build {
            audience,
            message: "GCP workload identity federation requires impersonate_service_account to \
                      be set"
                .to_owned(),
        });
    };

    let access_token_source = federated_access_token_source(sources, &wif_provider);
    let access_token_credentials = access_token_source
        .credentials
        .get_or_build(boxed_federated_access_token_source_build(
            wif_provider.clone(),
        ))
        .await
        .map_err(|message| GcpAuthError::Adc {
            audience: audience.clone(),
            impersonate: impersonate.clone(),
            message,
        })?;

    let credentials = idtoken::impersonated::Builder::from_source_credentials(
        audience.clone(),
        impersonate,
        access_token_credentials,
    )
    .build()
    .map_err(|e| GcpAuthError::Build {
        audience,
        message: e.to_string(),
    })?;

    Ok(Arc::new(FederatedIdTokenCredentials {
        credentials,
        _access_token_source: access_token_source,
    }) as Arc<dyn IdTokenSource>)
}

/// Atomic lookup-or-create under `sources`' lock: upgrades `provider`'s existing entry if it's
/// still live, or replaces it (whether absent or dead) with a freshly constructed
/// `Arc<FederatedAccessTokenSource>` and returns that. Concurrent cold constructions for one
/// provider still converge on one instance despite this running under a plain lock: the work done
/// while holding it is just `HashMap`/`Weak` bookkeeping and one cheap `RecoverableCell::new()` --
/// the potentially slow part (AWS credential resolution and Google STS construction) happens
/// later, inside `RecoverableCell`'s own single-flight, never while holding this lock.
pub(super) fn federated_access_token_source(
    sources: &FederatedAccessTokenSources,
    provider: &str,
) -> Arc<FederatedAccessTokenSource> {
    let mut sources = sources.lock();
    if let Some(existing) = sources.get(provider).and_then(Weak::upgrade) {
        return existing;
    }
    let fresh = Arc::new(FederatedAccessTokenSource {
        credentials: RecoverableCell::new(),
    });
    sources.insert(provider.to_owned(), Arc::downgrade(&fresh));
    fresh
}

/// Probes `provider`'s access-token source credentials and replaces them if -- and only if -- the
/// probe proves their background refresh task has permanently died, mirroring
/// `CredentialRegistry::recover_ambient_source_if_dead` exactly. Called from `mint()` after any
/// permanent mint failure on a federated key targeting `provider`.
///
/// Deliberately upgrades rather than looking up via [`federated_access_token_source`], and is a
/// no-op if the entry is absent or already dead: a dead weak entry here means no live outer
/// credential needs this provider's source any more (the caller's own `mint()` still holds one
/// strong reference for the duration of this call -- see the module-level recovery/reap race
/// invariant below -- so "dead" here can only mean genuinely unreferenced), and creating one
/// afresh only to immediately have nothing reference it would just hand
/// [`reap_disused_federated_access_token_sources`] a tombstone to prune next tick for no reason.
/// The next real outer construction creates a fresh source on its own via
/// [`federated_access_token_source`] regardless.
pub(super) async fn recover_federated_access_token_source_if_dead(
    sources: &FederatedAccessTokenSources,
    provider: &str,
) {
    let Some(access_token_source) = sources.lock().get(provider).and_then(Weak::upgrade) else {
        return;
    };
    match access_token_source
        .credentials
        .replace_if_failed(
            super::credentials_source_is_dead,
            boxed_federated_access_token_source_build(provider.to_owned()),
        )
        .await
    {
        Ok(true) => {
            // `provider_resource` is a log field, not a metric label -- provider cardinality is
            // unbounded from this crate's point of view, so it must never appear on a metric.
            warn!(
                provider_resource = %provider,
                "replaced a federated GCP access-token source: its refresh task was proven dead"
            );
        }
        Ok(false) => {}
        Err(error) => {
            warn!(
                provider_resource = %provider,
                error = %error,
                "failed to rebuild a federated GCP access-token source after its refresh task \
                 was proven dead; a future mint attempt will retry"
            );
        }
    }
}

/// Prunes `sources` of every entry whose weak reference no longer upgrades -- i.e. every provider
/// with no live [`FederatedIdTokenCredentials`] referencing it any more -- and returns the number
/// of entries retained (all upgradeable, hence live), for `gcp.federation.sources.active`.
///
/// Called from the registry's housekeeping tick, and only after the caller has
/// already driven the *outer* moka cache's own `run_pending_tasks`: moka evicts idle entries
/// lazily, so an outer credential can be logically expired yet still be a live strong referent of
/// its access-token source until that pass actually drops it. Pruning here first would remove
/// sources whose only referent simply hasn't been dropped yet, not sources that are actually
/// unreferenced -- and since this pass runs once per `CACHE_HOUSEKEEPING_INTERVAL`, an
/// already-unreferenced source can also sit here for up to one more interval before this catches
/// it; `gcp.federation.sources.active` is documented as approximate for exactly that reason.
pub(super) fn reap_disused_federated_access_token_sources(
    sources: &FederatedAccessTokenSources,
) -> usize {
    let mut sources = sources.lock();
    sources.retain(|_, weak| weak.strong_count() > 0);
    sources.len()
}

/// Boxes [`build_federated_access_token_source`]'s future. `RecoverableCell::get_or_build`/
/// `replace_if_failed` are small generic utilities that hold their `build` future inline across an
/// await point; `build_federated_access_token_source` resolves the shared AWS credentials
/// internally (`aws_config::load_defaults()`'s own future is large), so leaving it unboxed here would size
/// `mint()`'s own future -- which reaches this through both `build_federated_source` and recovery
/// -- for that on every mint, federated or not.
fn boxed_federated_access_token_source_build(
    provider_resource: String,
) -> std::pin::Pin<
    Box<dyn Future<Output = Result<google_cloud_auth::credentials::Credentials, String>> + Send>,
> {
    Box::pin(build_federated_access_token_source(provider_resource))
}

/// Test-only override for a federated access-token source's build step, keyed by provider so
/// distinct providers' tests cannot interfere with each other. Consulted before resolving AWS credentials
/// at all (see [`build_federated_access_token_source`]), so overriding tests never need
/// `[gcp-federation]` configuration or [`AwsFederationCredentials`] in place.
#[cfg(test)]
type FederatedAccessTokenSourceOverride =
    Arc<dyn Fn() -> Result<google_cloud_auth::credentials::Credentials, String> + Send + Sync>;
#[cfg(test)]
static FEDERATED_ACCESS_TOKEN_SOURCE_OVERRIDES: std::sync::LazyLock<
    parking_lot::Mutex<std::collections::HashMap<String, FederatedAccessTokenSourceOverride>>,
> = std::sync::LazyLock::new(Default::default);

/// Builds fresh access-token credentials for `provider`: the test override when compiled for
/// tests, otherwise the real chain (resolve the shared AWS credentials, sign a SigV4 subject token,
/// exchange it at Google STS). Shared by [`build_federated_source`] (first build) and
/// [`recover_federated_access_token_source_if_dead`] (rebuild after a proven-dead probe).
async fn build_federated_access_token_source(
    provider_resource: String,
) -> Result<google_cloud_auth::credentials::Credentials, String> {
    #[cfg(test)]
    if let Some(f) = FEDERATED_ACCESS_TOKEN_SOURCE_OVERRIDES
        .lock()
        .get(&provider_resource)
        .cloned()
    {
        return f();
    }

    let aws_federation_credentials = aws_federation_credentials().await?;
    let subject_token_provider = Arc::new(AwsSubjectTokenProvider {
        aws_federation_credentials,
        provider_resource: provider_resource.clone(),
    });
    ProgrammaticBuilder::new(subject_token_provider)
        .with_audience(provider_resource)
        .with_subject_token_type(AWS4_SUBJECT_TOKEN_TYPE)
        .with_token_url(GOOGLE_STS_TOKEN_URL)
        .build()
        .map_err(|e| format!("building GCP workload identity federation source credentials: {e}"))
}

/// Test-only: install `f` as the build step for `provider`'s shared access-token source (see
/// [`FEDERATED_ACCESS_TOKEN_SOURCE_OVERRIDES`]). `f` returns a
/// `google_cloud_auth::credentials::Credentials` directly -- a fully public type -- so callers
/// outside this module (e.g. `gcp::tests`) can install one without needing to see anything
/// internal to federation.
#[cfg(test)]
pub(super) fn install_federated_access_token_source_override_for_test(
    provider: &str,
    f: impl Fn() -> Result<google_cloud_auth::credentials::Credentials, String> + Send + Sync + 'static,
) {
    FEDERATED_ACCESS_TOKEN_SOURCE_OVERRIDES
        .lock()
        .insert(provider.to_owned(), Arc::new(f));
}

#[cfg(test)]
mod federation_tests {
    use std::time::{Duration, SystemTime};

    use aws_credential_types::Credentials as AwsCredentials;
    use google_cloud_auth::errors::SubjectTokenProviderError;

    use super::build_subject_token;

    const PROVIDER: &str = "//iam.googleapis.com/projects/123456789/locations/global/workloadIdentityPools/restate-cloud/providers/aws-federation";

    fn fixed_credentials() -> AwsCredentials {
        AwsCredentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("SESSIONTOKENEXAMPLE".to_owned()),
            None,
            "test",
        )
    }

    fn fixed_time() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_755_000_000)
    }

    #[test]
    fn subject_token_has_the_shape_google_sts_expects() {
        let token =
            build_subject_token(&fixed_credentials(), "us-east-1", PROVIDER, fixed_time()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&token).unwrap();

        assert_eq!(parsed["method"], "POST");
        assert_eq!(parsed["body"], "");
        assert_eq!(
            parsed["url"],
            "https://sts.us-east-1.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15"
        );

        let headers: Vec<(String, String)> = parsed["headers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| {
                (
                    h["key"].as_str().unwrap().to_lowercase(),
                    h["value"].as_str().unwrap().to_owned(),
                )
            })
            .collect();

        let names: Vec<&str> = headers.iter().map(|(k, _)| k.as_str()).collect();
        assert!(names.contains(&"authorization"), "{names:?}");
        assert!(names.contains(&"host"), "{names:?}");
        assert!(names.contains(&"x-amz-date"), "{names:?}");
        assert!(names.contains(&"x-amz-security-token"), "{names:?}");
        assert!(names.contains(&"x-goog-cloud-target-resource"), "{names:?}");
    }

    /// The provider resource name must be covered by the signature. If it were not, a signed
    /// envelope minted for one workload identity pool could be replayed against another -- which
    /// is the environment-isolation boundary this authentication scheme relies on.
    #[test]
    fn target_resource_is_a_signed_header() {
        let token =
            build_subject_token(&fixed_credentials(), "us-east-1", PROVIDER, fixed_time()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&token).unwrap();

        let authorization = parsed["headers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| {
                h["key"]
                    .as_str()
                    .unwrap()
                    .eq_ignore_ascii_case("authorization")
            })
            .expect("authorization header")["value"]
            .as_str()
            .unwrap()
            .to_owned();

        assert!(
            authorization.contains("x-goog-cloud-target-resource"),
            "target resource not in SignedHeaders: {authorization}"
        );
    }

    // -- Tests below exercise the real `AwsSubjectTokenProvider` + `ProgrammaticBuilder` chain
    // against a local mock Google STS server. They build a `AwsFederationCredentials` directly (bypassing
    // `AwsFederationCredentials::init`'s `aws_config::load_defaults()` and the process-global `AWS_FEDERATION_CREDENTIALS`/
    // `FEDERATION_CONFIG` statics entirely) with a pre-cached, never-expiring credentials
    // fixture, so `AwsFederationCredentials::credentials()` never calls the real `sts:AssumeRole` API -- no network
    // access to AWS is needed or attempted. This also means these tests never touch the shared
    // statics `aws_federation_credentials()` reads, so they cannot interfere with `gcp::tests`'
    // `wif_requested_without_server_config_is_a_permanent_build_error`, which depends on
    // `FEDERATION_CONFIG` staying uninstalled for the lifetime of this test binary.

    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use google_cloud_auth::credentials::CacheableResource;
    use google_cloud_auth::credentials::external_account::ProgrammaticBuilder;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    use super::{AWS4_SUBJECT_TOKEN_TYPE, AwsFederationCredentials, AwsSubjectTokenProvider};

    /// Builds pre-seeded AWS federation credentials that never perform network I/O.
    fn fixture_aws_federation_credentials() -> Arc<AwsFederationCredentials> {
        let credentials = fixed_credentials();
        Arc::new(AwsFederationCredentials {
            region: "us-east-1".to_owned(),
            provider: aws_credential_types::provider::SharedCredentialsProvider::new(
                credentials.clone(),
            ),
            cached: tokio::sync::Mutex::new(Some(credentials)),
        })
    }

    fn aws_env_snapshot() -> std::collections::BTreeMap<String, String> {
        std::env::vars()
            .filter(|(k, _)| k.starts_with("AWS_"))
            .collect()
    }

    type CapturedBody = Arc<Mutex<Option<Bytes>>>;

    /// Stand up a tiny local Google STS stand-in: captures the raw POST body of the first request
    /// and responds with a well-formed token-exchange response.
    async fn mock_google_sts() -> (String, CapturedBody) {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind mock STS server");
        let addr = listener.local_addr().expect("local_addr");
        let captured: CapturedBody = Arc::new(Mutex::new(None));
        let captured_for_task = Arc::clone(&captured);

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => return,
                };
                let captured = Arc::clone(&captured_for_task);
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let captured = Arc::clone(&captured);
                        async move {
                            let body = req.into_body().collect().await.unwrap().to_bytes();
                            *captured.lock().unwrap() = Some(body);
                            let response_body = serde_json::json!({
                                "access_token": "mock-federated-access-token",
                                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                                "token_type": "Bearer",
                                "expires_in": 3600,
                            })
                            .to_string();
                            Ok::<_, Infallible>(
                                Response::builder()
                                    .status(StatusCode::OK)
                                    .header("content-type", "application/json")
                                    .body(Full::new(Bytes::from(response_body)))
                                    .expect("response build"),
                            )
                        }
                    });
                    let _ = http1::Builder::new().serve_connection(io, svc).await;
                });
            }
        });

        (format!("http://{addr}/token"), captured)
    }

    /// Exercises the real `AwsSubjectTokenProvider` + `ProgrammaticBuilder` path against a mock
    /// Google STS: the exchange request's `grant_type`/`subject_token_type`/`audience` fields, that
    /// the AIP-4117 JSON arrives at Google percent-encoded exactly once after undoing the STS
    /// exchange's own transport form-encoding, and that none of it touches any `AWS_*` environment
    /// variable. Stops at the STS hop: `idtoken::impersonated::Builder`'s impersonation URL has no
    /// public override in google-cloud-auth 1.15, so `generateIdToken` cannot be redirected to a
    /// local mock from outside the crate.
    #[restate_core::test]
    async fn programmatic_builder_sts_exchange_matches_expected_wire_format() {
        let env_before = aws_env_snapshot();

        let aws_federation_credentials = fixture_aws_federation_credentials();
        let subject_token_provider = Arc::new(AwsSubjectTokenProvider {
            aws_federation_credentials,
            provider_resource: PROVIDER.to_owned(),
        });

        let (token_url, captured) = mock_google_sts().await;
        let credentials = ProgrammaticBuilder::new(subject_token_provider)
            .with_audience(PROVIDER)
            .with_subject_token_type(AWS4_SUBJECT_TOKEN_TYPE)
            .with_token_url(token_url)
            .build()
            .expect("programmatic external-account credentials build");

        let result = credentials.headers(http::Extensions::new()).await;
        let CacheableResource::New { data: headers, .. } =
            result.expect("STS exchange succeeds against the mock server")
        else {
            panic!("expected fresh headers on first fetch");
        };
        assert!(
            headers.contains_key(http::header::AUTHORIZATION),
            "exchanged credentials should produce an Authorization header: {headers:?}"
        );

        let body = captured
            .lock()
            .unwrap()
            .clone()
            .expect("mock STS server received a request");
        let form: std::collections::HashMap<String, String> =
            url::form_urlencoded::parse(&body).into_owned().collect();

        assert_eq!(
            form.get("grant_type").map(String::as_str),
            Some("urn:ietf:params:oauth:grant-type:token-exchange")
        );
        assert_eq!(
            form.get("subject_token_type").map(String::as_str),
            Some(AWS4_SUBJECT_TOKEN_TYPE)
        );
        assert_eq!(form.get("audience").map(String::as_str), Some(PROVIDER));

        // The subject_token field on the wire has been through two encoding layers: ours (this
        // module's `byte_serialize` over the SigV4 envelope) and the STS exchange's own
        // form-encoding for transport. `url::form_urlencoded::parse` above already undid the
        // transport layer; recomputing our own encoding step independently and comparing proves
        // what arrived is encoded exactly once, not zero or two times.
        let subject_token_wire = form
            .get("subject_token")
            .expect("subject_token field present");
        let envelope = build_subject_token(
            &fixed_credentials(),
            "us-east-1",
            PROVIDER,
            SystemTime::now(),
        )
        .expect("rebuilding an envelope with the same shape succeeds");
        // Time-dependent (SigV4 signs the request time), so compare shape rather than equality:
        // both must be a once-encoded JSON envelope with the same structural fields.
        let decoded_wire: String = url::form_urlencoded::parse(subject_token_wire.as_bytes())
            .next()
            .map(|(k, _)| k.into_owned())
            .expect("subject token decodes to one key");
        let decoded_wire_json: serde_json::Value =
            serde_json::from_str(&decoded_wire).expect("decoded subject token is JSON");
        let envelope_json: serde_json::Value =
            serde_json::from_str(&envelope).expect("locally-built envelope is JSON");
        assert_eq!(decoded_wire_json["method"], envelope_json["method"]);
        assert_eq!(decoded_wire_json["url"], envelope_json["url"]);

        assert_eq!(
            env_before,
            aws_env_snapshot(),
            "constructing and using a federated subject-token provider must not mutate any \
             AWS_* environment variable"
        );
    }

    #[test]
    fn accepts_a_well_formed_aws_role_arn() {
        super::validate_aws_role_arn("arn:aws:iam::123456789012:role/RestateCloudGcpFederation")
            .expect("well-formed ARN accepted");
        super::validate_aws_role_arn("arn:aws-us-gov:iam::123456789012:role/path/to/role")
            .expect("non-default partition with a role path accepted");
    }

    #[test]
    fn rejects_malformed_aws_role_arns() {
        for arn in [
            "not-an-arn",
            "arn:aws:s3::123456789012:role/wrong-service",
            "arn:aws:iam:us-east-1:123456789012:role/has-a-region",
            "arn:aws:iam::12345:role/too-short-account",
            "arn:aws:iam::12345678901a:role/non-numeric-account",
            "arn:aws:iam::123456789012:user/not-a-role",
            "arn:aws:iam::123456789012:role/",
            "arn:aws!:iam::123456789012:role/bad-partition-chars",
        ] {
            super::validate_aws_role_arn(arn)
                .expect_err(&format!("expected '{arn}' to be rejected"));
        }
    }

    #[test]
    fn accepts_well_formed_session_names() {
        for name in [
            "ab",
            "env-xyz",
            "env_xyz.123@foo,bar+baz=qux",
            &"a".repeat(64),
        ] {
            super::validate_aws_role_session_name(name).expect("well-formed session name accepted");
        }
    }

    #[test]
    fn rejects_malformed_session_names() {
        for name in [
            "",
            "a",
            &"a".repeat(65),
            "has a space",
            "has/slash",
            "has#hash",
        ] {
            super::validate_aws_role_session_name(name)
                .expect_err(&format!("expected '{name}' to be rejected"));
        }
    }

    #[test]
    fn install_config_rejects_invalid_aws_role_arn() {
        let config = super::GcpFederationOptions {
            aws_role_arn: "not-an-arn".to_owned(),
            aws_role_session_name: "valid-session".to_owned(),
        };
        super::install_config(Some(config)).expect_err("invalid aws-role-arn must be rejected");
    }

    #[test]
    fn install_config_rejects_invalid_session_name() {
        let config = super::GcpFederationOptions {
            aws_role_arn: "arn:aws:iam::123456789012:role/RestateCloudGcpFederation".to_owned(),
            aws_role_session_name: "has a space".to_owned(),
        };
        super::install_config(Some(config))
            .expect_err("invalid aws-role-session-name must be rejected");
    }

    fn fixture_config(aws_role_arn: &str) -> super::GcpFederationOptions {
        super::GcpFederationOptions {
            aws_role_arn: aws_role_arn.to_owned(),
            aws_role_session_name: "session".to_owned(),
        }
    }

    /// Exercise config comparison without mutating the process-global cell.
    #[test]
    fn decide_config_install_outcomes() {
        let a = fixture_config("arn:aws:iam::123456789012:role/A");
        let b = fixture_config("arn:aws:iam::123456789012:role/B");

        assert_eq!(
            super::decide_config_install(None, None),
            super::ConfigInstallOutcome::NotRequested
        );
        assert_eq!(
            super::decide_config_install(None, Some(&a)),
            super::ConfigInstallOutcome::FirstInstall
        );
        assert_eq!(
            super::decide_config_install(Some(&a), Some(&a.clone())),
            super::ConfigInstallOutcome::Unchanged
        );
        assert_eq!(
            super::decide_config_install(Some(&a), Some(&b)),
            super::ConfigInstallOutcome::Differing
        );
        assert_eq!(
            super::decide_config_install(Some(&a), None),
            super::ConfigInstallOutcome::RemovalIgnored
        );
    }

    #[test]
    fn describe_config_diff_names_only_the_changed_field() {
        let a = fixture_config("arn:aws:iam::123456789012:role/A");
        let mut session_changed = a.clone();
        session_changed.aws_role_session_name = "different-session".to_owned();

        let diff = super::describe_config_diff(&a, &session_changed);
        assert!(
            diff.contains("aws-role-session-name") && !diff.contains("aws-role-arn"),
            "an aws-role-session-name-only change must not report aws-role-arn: {diff}"
        );

        let b = fixture_config("arn:aws:iam::123456789012:role/B");
        let diff = super::describe_config_diff(&a, &b);
        assert!(
            diff.contains("aws-role-arn") && !diff.contains("aws-role-session-name"),
            "an aws-role-arn-only change must not report aws-role-session-name: {diff}"
        );
    }

    /// A differing live-reload must keep the installed value without failing the caller. This
    /// runs in its own nextest process because it mutates process-global configuration.
    #[test]
    fn install_config_logs_and_keeps_the_original_on_a_differing_reinstall() {
        super::install_config(Some(fixture_config(
            "arn:aws:iam::123456789012:role/Original",
        )))
        .expect("first install succeeds");
        super::install_config(Some(fixture_config(
            "arn:aws:iam::123456789012:role/Different",
        )))
        .expect("a differing re-install must not fail the caller");
    }

    /// An invalid replacement is still ignored: validating a value that cannot take effect would
    /// fail the caller without changing the active configuration.
    #[test]
    fn install_config_ignores_an_invalid_differing_reinstall() {
        super::install_config(Some(fixture_config(
            "arn:aws:iam::123456789012:role/Original",
        )))
        .expect("first install succeeds");

        super::install_config(Some(super::GcpFederationOptions {
            aws_role_arn: "not-an-arn".to_owned(),
            aws_role_session_name: "session".to_owned(),
        }))
        .expect("an invalid differing reload must not fail the caller");

        assert_eq!(
            super::FEDERATION_CONFIG
                .get()
                .map(|c| c.aws_role_arn.as_str()),
            Some("arn:aws:iam::123456789012:role/Original"),
            "the original config must remain installed after an invalid differing reload"
        );
    }

    /// Removing the block on live reload is ignored like any other unsupported change.
    #[test]
    fn install_config_ignores_a_block_removed_on_reload() {
        super::install_config(Some(fixture_config(
            "arn:aws:iam::123456789012:role/Original",
        )))
        .expect("first install succeeds");

        super::install_config(None).expect("a removed block must not fail the caller");

        assert_eq!(
            super::FEDERATION_CONFIG
                .get()
                .map(|c| c.aws_role_arn.as_str()),
            Some("arn:aws:iam::123456789012:role/Original"),
            "the original config must remain installed after the block is removed on reload"
        );
    }

    fn assume_role_access_denied_error() -> aws_credential_types::provider::error::CredentialsError
    {
        use aws_sdk_sts::error::SdkError;
        use aws_sdk_sts::operation::assume_role::AssumeRoleError;
        use aws_smithy_runtime_api::client::orchestrator::HttpResponse;
        use aws_smithy_runtime_api::http::StatusCode;
        use aws_smithy_types::body::SdkBody;
        use aws_smithy_types::error::ErrorMetadata;

        let service_error = AssumeRoleError::generic(
            ErrorMetadata::builder()
                .code("AccessDenied")
                .message(
                    "User: arn:aws:sts::123456789012:assumed-role/... is not authorized to \
                     perform: sts:AssumeRole on resource: ...",
                )
                .build(),
        );
        let raw = HttpResponse::new(
            StatusCode::try_from(403).unwrap(),
            SdkBody::from("<AccessDenied/>"),
        );
        aws_credential_types::provider::error::CredentialsError::provider_error(
            SdkError::service_error(service_error, raw),
        )
    }

    fn assume_role_dispatch_timeout_error()
    -> aws_credential_types::provider::error::CredentialsError {
        use aws_sdk_sts::error::SdkError;
        use aws_sdk_sts::operation::assume_role::AssumeRoleError;

        let sdk_error: SdkError<AssumeRoleError> =
            SdkError::timeout_error("connect timed out reaching sts.us-east-1.amazonaws.com");
        aws_credential_types::provider::error::CredentialsError::provider_error(sdk_error)
    }

    #[test]
    fn assume_role_access_denied_classifies_permanent() {
        let raw = assume_role_access_denied_error();
        // The wire error code is recovered through the unmodeled `Unhandled` variant, not matched
        // on `Display` text -- pin that alongside the classification it feeds.
        assert_eq!(super::assume_role_error_code(&raw), Some("AccessDenied"));
        let error = super::federation_error_from_assume_role_failure(&raw);
        assert!(
            !error.is_transient(),
            "an AssumeRole AccessDenied must classify as permanent, got {error:?}"
        );
    }

    #[test]
    fn assume_role_dispatch_timeout_classifies_transient() {
        let raw = assume_role_dispatch_timeout_error();
        assert_eq!(super::assume_role_error_code(&raw), None);
        let error = super::federation_error_from_assume_role_failure(&raw);
        assert!(
            error.is_transient(),
            "an AssumeRole connector/timeout failure must classify as transient, got {error:?}"
        );
    }

    /// `federated_access_token_sources` is weak-indexed, not time-evicted (see that type's doc):
    /// while a provider's `FederatedAccessTokenSource` is live, every lookup must upgrade to the
    /// exact same instance -- never rebuild. Once every strong reference to it drops (simulated
    /// here directly, without any outer credential involved), the entry is a dead tombstone the
    /// *next* lookup must replace with a fresh instance, never resurrect. Housekeeping's own
    /// pruning of that tombstone is covered separately, in `gcp::tests`, through the full
    /// production construction path.
    #[test]
    fn federated_access_token_source_upgrades_while_live_and_replaces_once_dead() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/lookup-test";
        let sources: super::FederatedAccessTokenSources = Default::default();

        let first = super::federated_access_token_source(&sources, provider);
        let second = super::federated_access_token_source(&sources, provider);
        assert!(
            Arc::ptr_eq(&first, &second),
            "a live provider must always resolve to the same access-token source"
        );

        let dead_ptr = Arc::as_ptr(&first);
        drop(first);
        drop(second);
        let third = super::federated_access_token_source(&sources, provider);
        assert_ne!(
            Arc::as_ptr(&third),
            dead_ptr,
            "once every strong reference drops, the next lookup must replace the tombstone with \
             a fresh instance, never resurrect the dead one"
        );
    }
}
