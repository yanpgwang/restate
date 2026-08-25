// Copyright (c) 2023 - 2026 Restate Software, Inc., Restate GmbH.
// All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

//! GCP OIDC ID-token mint client for HTTP deployments hosted on Cloud Run and similar
//! Google-fronted endpoints.
//!
//! Building a `google-cloud-auth` credential starts a background refresh task that lives as long as
//! the credential does. The entire credential is cached in a process-global registry keyed by
//! `(audience, impersonate_service_account)`; impersonated keys additionally share a process-wide
//! ambient source credential (see [`CredentialRegistry::ambient_source`]). Construction runs as a
//! [`TaskKind::Credentials`] task on TaskCenter's default runtime, so each credential's refresh task
//! lands on a runtime with process lifetime rather than the runtime that happens to call `mint()`.
//!
//! [`GcpTokenClient`] captures its [`Handle`] explicitly at construction and threads it through to
//! every registry lookup and credential-build spawn; it never reads a `TaskCenter` task-local.
//! `mint()` is therefore safe to call from an ordinary `tokio::spawn` task with no task-locals of
//! its own -- which is exactly how invoker invocation tasks run, on a plain `tokio::JoinSet`.
//!
//! A deployment may additionally request AWS -> GCP workload identity federation (see
//! [`federation`]): rather than ambient Application Default Credentials, the ID token is minted
//! through a shared AWS broker role, a SigV4-signed subject token, a Google STS exchange, and
//! impersonation. That path is a fourth [`IdTokenSource`] construction recipe, keyed by
//! `wif_provider` and cached and refreshed exactly like the others; unlike them, its construction
//! is pure async I/O (no blocking ADC reads), so it dispatches from the same
//! [`CredentialRegistry::build_on_tc_task`] task without going through the blocking-build path.

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use metrics::{counter, gauge, histogram};
use moka::future::Cache;
use moka::ops::compute::Op;
use parking_lot::RwLock;
use restate_core::{Handle, TaskKind};
use thiserror::Error;
use tokio::sync::Semaphore;
use tracing::warn;

use crate::metric_definitions::{
    GCP_CREDENTIAL_BUILD_DURATION, GCP_CREDENTIAL_BUILDS, GCP_CREDENTIALS_ACTIVE, GCP_TOKEN_MINTS,
    MINT_MODE_ADC, MINT_MODE_FEDERATED, MINT_OUTCOME_BUILD_ERROR, MINT_OUTCOME_PERMANENT_ERROR,
    MINT_OUTCOME_SUCCESS, MINT_OUTCOME_TIMEOUT, MINT_OUTCOME_TRANSIENT_ERROR, RESULT_ERROR,
    RESULT_SUCCESS,
};

#[cfg(any(test, feature = "test_util"))]
use ahash::HashMap;
#[cfg(any(test, feature = "test_util"))]
use parking_lot::Mutex;

mod federation;

pub(crate) use federation::install_config as install_federation_config;

const MINT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);

const CACHE_TIME_TO_IDLE: Duration = Duration::from_secs(3600);

/// moka evicts lazily; drive eviction when no mint attempts are happening
const CACHE_HOUSEKEEPING_INTERVAL: Duration = Duration::from_secs(300);

/// Bound on probing a shared source credential's own cached state (see
/// [`credentials_source_is_dead`]). Shared by the ambient source and, per WIF provider, the
/// federated access-token source.
const SOURCE_PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Caps concurrent blocking `google-cloud-auth`/ADC builds, so a burst of distinct new keys (or a
/// GCP outage causing every retry to rebuild) cannot exhaust tokio's blocking thread pool. This is
/// fixed rather than CPU-scaled because Tokio's blocking pool is independent of CPU concurrency.
/// The
/// federated build path is pure async I/O (no blocking calls), so it is deliberately not routed
/// through this bound -- see [`CredentialRegistry::build_on_tc_task`].
const MAX_CONCURRENT_BLOCKING_BUILDS: usize = 4;

#[derive(Clone, Debug, Error)]
pub enum GcpAuthError {
    #[error(
        "failed to load Application Default Credentials (audience '{audience}', impersonating '{impersonate}'): {message}"
    )]
    Adc {
        audience: String,
        impersonate: String,
        message: String,
    },
    #[error("failed to build ID token credentials for audience '{audience}': {message}")]
    Build { audience: String, message: String },
    #[error(
        "the ambient Application Default Credentials identity cannot mint an ID token for audience '{audience}'. \
         User credentials (from `gcloud auth application-default login`) and Workload Identity Federation \
         (`external_account`) sources cannot mint ID tokens directly; set `--gcp-impersonate-service-account` to \
         mint the token via impersonation, or run Restate with a service-account key (`GOOGLE_APPLICATION_CREDENTIALS`) \
         or a GCE/GKE/Cloud Run metadata-server identity"
    )]
    AmbientUnsupported { audience: String },
    #[error(
        "failed to mint ID token (audience '{audience}', impersonating '{impersonate}'): {message}"
    )]
    Mint {
        audience: String,
        impersonate: String,
        message: String,
    },
    #[error(
        "token mint timed out after {duration:?} (audience '{audience}', impersonating '{impersonate}')"
    )]
    Timeout {
        audience: String,
        impersonate: String,
        duration: Duration,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct IdTokenSpec {
    /// Full resource name of a GCP workload identity federation provider. `None` selects the
    /// ambient/impersonated ADC paths, byte-for-byte as before workload identity federation
    /// existed; `Some` selects the federation chain in [`federation::build_federated_source`].
    wif_provider: Option<String>,
    impersonate: Option<String>,
    audience: String,
}

/// Renders `error`'s `Display` together with its full `source()` chain, each level separated by
/// `": "`.
///
/// `google_cloud_auth`'s `CredentialsError::Display` only ever prints its own top-level message
/// (e.g. "failed to fetch ID token via impersonation and future attempts will not succeed") --
/// the detail an operator actually needs, such as a `google_cloud_gax::error::Error` carrying the
/// HTTP status code and response body from a failed `iamcredentials.googleapis.com` call (e.g.
/// "the HTTP transport reports a [403] error: {"error":{"code":403,"status":"PERMISSION_DENIED",
/// ...}}"), lives one or more levels down its `source()` chain, which `Display` never visits.
/// Without this, a missing `roles/iam.serviceAccountOpenIdTokenCreator` binding -- the single most
/// common federation/impersonation misconfiguration -- surfaced as an opaque, undiagnosable
/// message in both `restate dp register` errors and server logs.
fn display_error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(err) = cause {
        message.push_str(": ");
        message.push_str(&err.to_string());
        cause = err.source();
    }
    message
}

/// Internal seam that keeps the registry and the mint path testable without ADC or network: `Live`
/// wraps a real `IDTokenCredentials`, while tests inject mocks and a test-only `MockToken` source
/// (see [`GcpTokenClient::seed_for_test`]).
#[async_trait]
trait IdTokenSource: Send + Sync {
    async fn id_token(&self) -> Result<String, google_cloud_auth::errors::CredentialsError>;
}

struct Live(google_cloud_auth::credentials::idtoken::IDTokenCredentials);

#[async_trait]
impl IdTokenSource for Live {
    async fn id_token(&self) -> Result<String, google_cloud_auth::errors::CredentialsError> {
        self.0.id_token().await
    }
}

/// Credential registry: a cache of credential objects and the shared ambient credentials source,
/// tied to the [`Handle`] that spawned its background housekeeping task. Every [`GcpTokenClient`]
/// reaches its registry through [`credential_registry`], passing its own stored `Handle`; the
/// registry is rebuilt whenever that handle differs from the one it was built under -- see that
/// function's doc comment.
struct CredentialRegistry {
    self_weak: Weak<CredentialRegistry>,
    task_center: Handle,
    cache: Cache<IdTokenSpec, Arc<dyn IdTokenSource>>,
    ambient_source: RecoverableCell<google_cloud_auth::credentials::Credentials>,
    /// Per-provider federated access-token sources, weak-indexed and leased by every cached outer
    /// federated ID-token credential -- see [`federation::FederatedAccessTokenSources`] for the
    /// ownership model.
    federated_access_token_sources: federation::FederatedAccessTokenSources,
    #[cfg(any(test, feature = "test_util"))]
    test_hooks: TestHooks,
}

/// A single cached, cheaply-cloneable value with single-flight, retry-on-error construction, and
/// supporting replacement in case the stored value is later found to be failed. Holds one value; a
/// caller keying by provider identity (federation keys by WIF provider resource name; see
/// [`federation::federated_access_token_source`]) is the caller's job, not this type's.
struct RecoverableCell<T> {
    cell: tokio::sync::Mutex<Option<T>>,
}

impl<T: Clone> RecoverableCell<T> {
    fn new() -> Self {
        Self {
            cell: tokio::sync::Mutex::new(None),
        }
    }

    async fn get_or_build<E>(&self, build: impl Future<Output = Result<T, E>>) -> Result<T, E> {
        let mut guard = self.cell.lock().await;
        if let Some(value) = guard.as_ref() {
            return Ok(value.clone());
        }
        let value = build.await?;
        *guard = Some(value.clone());
        Ok(value)
    }

    // bounded by the probe's own timeout; swapped under lock so no need for external ABA checks.
    // Returns whether a replacement actually happened, so callers can log accordingly.
    async fn replace_if_failed<E>(
        &self,
        should_replace: impl AsyncFnOnce(&T) -> bool,
        build: impl Future<Output = Result<T, E>>,
    ) -> Result<bool, E> {
        let mut guard = self.cell.lock().await;
        let Some(current) = guard.as_ref() else {
            return Ok(false);
        };
        if !should_replace(current).await {
            return Ok(false);
        }
        match build.await {
            Ok(value) => {
                *guard = Some(value);
                Ok(true)
            }
            Err(e) => {
                *guard = None;
                Err(e)
            }
        }
    }

    #[cfg(test)]
    async fn seed_for_test(&self, value: T) {
        *self.cell.lock().await = Some(value);
    }
}

#[cfg(any(test, feature = "test_util"))]
type ConstructOverride =
    Arc<dyn Fn(&IdTokenSpec) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> + Send + Sync>;
#[cfg(any(test, feature = "test_util"))]
type AmbientSourceOverride =
    Arc<dyn Fn() -> Result<google_cloud_auth::credentials::Credentials, String> + Send + Sync>;

/// Lets unit tests drive the real moka-backed construction/eviction paths without ADC or network.
/// One `TestHooks` per `CredentialRegistry`, so each test's own `TaskCenter` (see
/// `#[restate_core::test]` and `credential_registry`'s doc comment) gets its own overrides: tests
/// sharing a process under `nextest` do not interfere with each other even without process
/// isolation. `build_overrides` is further keyed by `IdTokenSpec` so distinct-key tests within one
/// registry don't interfere either; `ambient_source_override` is a single slot, since the ambient
/// source itself is a single shared value per registry.
#[cfg(any(test, feature = "test_util"))]
#[derive(Default)]
struct TestHooks {
    build_overrides: Mutex<HashMap<IdTokenSpec, ConstructOverride>>,
    ambient_source_override: Mutex<Option<AmbientSourceOverride>>,
}

struct RegistrySlot {
    registry: Arc<CredentialRegistry>,
    task_center: Handle,
}

static REGISTRY: RwLock<Option<RegistrySlot>> = RwLock::new(None);

/// Lives inside the registry's housekeeping future. `TaskKind::Credentials` is `OnCancel =
/// "abort"`, so TaskCenter shutdown drops that future (running this type's `Drop`) without
/// otherwise touching `REGISTRY`; without this, a stopped TaskCenter's registry -- its cache, its
/// federated sources, their refresh tasks, and the strong `Handle` pinning the dead TaskCenter
/// alive -- would sit in the slot until the next `credential_registry()` call for a *different*
/// TaskCenter happened to overwrite it.
///
/// `Drop` clears the slot only if it still holds *this* registry (compared by pointer via the
/// same `Weak` this type carries), so a guard whose registry has already been superseded is a
/// no-op rather than clobbering its successor. Sync-only and lock-only, no `.await`: `Drop` runs
/// during future teardown, which cannot poll further futures.
struct ClearRegistrySlotOnDrop {
    registry: Weak<CredentialRegistry>,
}

impl Drop for ClearRegistrySlotOnDrop {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut slot = REGISTRY.write();
        if matches!(slot.as_ref(), Some(current) if Arc::ptr_eq(&current.registry, &registry)) {
            *slot = None;
        }
    }
}

/// Returns the credential registry for `task_center`, building a fresh one on first use or
/// whenever `task_center` differs from the one the cached registry was built under. Every caller
/// passes in its own stored [`Handle`] rather than reading a task-local: `GcpTokenClient::mint()`
/// is reachable from invoker invocation tasks, which run on a plain `tokio::JoinSet` with no
/// `TaskCenter` task-local of their own, so this module never calls `TaskCenter::current()` on
/// any production path.
///
/// Embedded Restate creates and destroys task centers within one process (see
/// `Restate::create`/`Restate::stop`); a registry's cache, ambient source, and housekeeping task
/// are only meaningful for the task center that spawned that housekeeping task; that task center
/// shutting down cancels it and orphans the registry, so a later `mint()` under a *new* task
/// center must get a fresh registry, not the stale one. The fast (same task center) path costs
/// one read-lock and one pointer comparison.
///
/// If `task_center` differs from the slot's occupant and is itself shutting down, this does not
/// build and install a new registry: a stale client retained across an embedded-server restart
/// would otherwise recreate the old generation's registry and displace whatever the new
/// generation has already installed. That case returns a build-error message instead.
///
/// [`FEDERATION_CONFIG`](federation) and federation's `BROKER` stay plain process-wide statics
/// rather than moving here: `FEDERATION_CONFIG` is install-once operator configuration for the
/// process, not state tied to any one task center's runtime, and `BROKER` holds AWS credential
/// state (an `AssumeRoleProvider` plus a cached session) with no background refresh task of its
/// own -- neither has anything spawned onto a task center that could go stale when this registry
/// rebuilds. `federated_access_token_sources` is different: each source's refresh task is spawned
/// on whichever task center built it, so it must rebuild -- not persist -- across a replacement
/// too.
fn credential_registry(task_center: &Handle) -> Result<Arc<CredentialRegistry>, String> {
    if let Some(slot) = REGISTRY.read().as_ref()
        && slot.task_center.ptr_eq(task_center)
    {
        return Ok(slot.registry.clone());
    }

    let mut guard = REGISTRY.write();
    if let Some(slot) = guard.as_ref()
        && slot.task_center.ptr_eq(task_center)
    {
        return Ok(slot.registry.clone());
    }

    if task_center.is_shutdown_requested() {
        return Err("TaskCenter is shutting down".to_owned());
    }

    let registry = CredentialRegistry::init(task_center);
    *guard = Some(RegistrySlot {
        registry: registry.clone(),
        task_center: task_center.clone(),
    });
    Ok(registry)
}

impl CredentialRegistry {
    fn init(task_center: &Handle) -> Arc<Self> {
        Arc::new_cyclic(|self_weak| {
            crate::metric_definitions::describe_metrics();

            let cache = Cache::builder().time_to_idle(CACHE_TIME_TO_IDLE).build();

            let housekeeping_cache = cache.clone();
            let housekeeping_self_weak = self_weak.clone();
            let clear_slot_on_drop = ClearRegistrySlotOnDrop {
                registry: self_weak.clone(),
            };
            let housekeeping = async move {
                // Held for the lifetime of this future, not polled -- its only job is running its
                // Drop impl when TaskCenter shutdown aborts this task (see the type's doc comment).
                let _clear_slot_on_drop = clear_slot_on_drop;
                let mut interval = tokio::time::interval(CACHE_HOUSEKEEPING_INTERVAL);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    // Order matters: the outer cache's own pending-eviction pass must run FIRST.
                    // moka evicts idle entries lazily, so an outer federated credential can be
                    // logically expired yet still be a live strong referent of its access-token
                    // source until this drops it. Reaping access-token sources before this would
                    // prune sources whose only referent simply hasn't been dropped yet.
                    housekeeping_cache.run_pending_tasks().await;
                    gauge!(GCP_CREDENTIALS_ACTIVE).set(housekeeping_cache.entry_count() as f64);
                    if let Some(registry) = housekeeping_self_weak.upgrade() {
                        let live = federation::reap_disused_federated_access_token_sources(
                            &registry.federated_access_token_sources,
                        );
                        gauge!(crate::metric_definitions::GCP_FEDERATION_SOURCES_ACTIVE)
                            .set(live as f64);
                    }
                }
            };

            // Managed (not unmanaged) so TaskCenter shutdown cancels and awaits it -- an
            // unmanaged infinite loop races test-process teardown and shows up as a nextest leak.
            let _ = task_center.spawn(
                TaskKind::Credentials,
                "gcp-credential-housekeeping",
                housekeeping,
            );

            Self {
                self_weak: self_weak.clone(),
                task_center: task_center.clone(),
                cache,
                ambient_source: RecoverableCell::new(),
                federated_access_token_sources: federation::FederatedAccessTokenSources::default(),
                #[cfg(any(test, feature = "test_util"))]
                test_hooks: TestHooks::default(),
            }
        })
    }

    /// An owned handle to this registry, for moving into a spawned task that must outlive `&self`
    /// (see [`Self::build_on_tc_task`]). Always upgradeable: every live `CredentialRegistry` is
    /// reachable only through the [`Arc`] this weak reference was cloned from.
    fn arc(&self) -> Arc<Self> {
        self.self_weak
            .upgrade()
            .expect("a CredentialRegistry is always held by the Arc it was constructed in")
    }

    // concurrent misses for the same key coalesce, with waiters seeing the same error; failures are
    // not cached
    async fn get_or_build(
        &self,
        spec: &IdTokenSpec,
    ) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
        let result = self
            .cache
            .try_get_with_by_ref(spec, self.build_on_tc_task(spec.clone()))
            .await
            .map_err(|error| (*error).clone());
        if result.is_ok() {
            gauge!(GCP_CREDENTIALS_ACTIVE).set(self.cache.entry_count() as f64);
        }
        result
    }

    // guards against a slow caller evicting a freshly rebuilt healthy credential
    async fn evict_if_unchanged(&self, spec: &IdTokenSpec, expected: &Arc<dyn IdTokenSource>) {
        self.cache
            .entry_by_ref(spec)
            .and_compute_with(|entry| {
                let op = match &entry {
                    Some(entry) if Arc::ptr_eq(entry.value(), expected) => Op::Remove,
                    _ => Op::Nop,
                };
                std::future::ready(op)
            })
            .await;
    }

    async fn ambient_source(&self) -> Result<google_cloud_auth::credentials::Credentials, String> {
        self.ambient_source
            .get_or_build(self.build_ambient_source())
            .await
    }

    /// After a permanent *impersonated* mint failure -- which doesn't say whether the shared
    /// source died or just this key's target service account is misconfigured -- replace the
    /// source only if a probe proves its refresh task is gone; a misconfigured target with a
    /// healthy source (the common case) leaves it untouched, or every retry would strand a live
    /// refresh task.
    async fn recover_ambient_source_if_dead(&self) {
        match self
            .ambient_source
            .replace_if_failed(credentials_source_is_dead, self.build_ambient_source())
            .await
        {
            Ok(true) => {
                warn!(
                    "replaced the shared ambient GCP credential source: its refresh task was proven dead"
                );
            }
            Ok(false) => {}
            Err(error) => {
                warn!(
                    error = %error,
                    "failed to rebuild the ambient GCP credential source after its refresh task \
                     was proven dead; a future mint attempt will retry"
                );
            }
        }
    }

    /// Builds the ambient source credential, consulting the test override when compiled for
    /// tests. Spawned on `self.task_center` -- the `Handle` the registry was built under, not a
    /// task-local -- so the source's own refresh task lands on that `TaskCenter`'s default
    /// runtime regardless of which runtime is polling `mint()`.
    async fn build_ambient_source(
        &self,
    ) -> Result<google_cloud_auth::credentials::Credentials, String> {
        #[cfg(any(test, feature = "test_util"))]
        {
            let override_fn = self.test_hooks.ambient_source_override.lock().clone();
            if let Some(f) = override_fn {
                return f();
            }
        }

        let build = || {
            google_cloud_auth::credentials::Builder::default()
                .build()
                .map_err(|e| e.to_string())
        };
        let task = self
            .task_center
            .spawn_unmanaged(TaskKind::Credentials, "gcp-credential-build", async move {
                spawn_bounded_blocking(build).await
            })
            .map_err(|_| "TaskCenter is shutting down".to_owned())?;
        match task.await {
            Ok(Ok(result)) => result,
            Ok(Err(join_error)) => Err(format!("construction task panicked: {join_error}")),
            Err(_shutdown) => Err("GCP credential construction task failed".to_owned()),
        }
    }

    /// Builds the outer credential for `spec` as a single [`TaskKind::Credentials`] task on
    /// `self.task_center` -- the `Handle` the registry was built under -- so any refresh task
    /// `build()` spawns internally lands on a runtime with process lifetime, not the runtime on
    /// which `mint()` gets called. The test-override consult happens *inside* the spawned task
    /// (not in [`Self::get_or_build`]) so mock-backed construction tests exercise this same
    /// dispatch, rather than bypassing it -- true for both the ambient/impersonated and the
    /// federated arm below.
    async fn build_on_tc_task(
        &self,
        spec: IdTokenSpec,
    ) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
        let registry = self.arc();
        // The spawned task needs ownership of `spec`; one audience copy stays behind for the two
        // error paths, which consume it lazily.
        let audience = spec.audience.clone();
        let task = self
            .task_center
            .spawn_unmanaged(TaskKind::Credentials, "gcp-credential-build", async move {
                #[cfg(any(test, feature = "test_util"))]
                if let Some(f) = registry
                    .test_hooks
                    .build_overrides
                    .lock()
                    .get(&spec)
                    .cloned()
                {
                    return f(&spec);
                }
                let start = Instant::now();
                let result = if spec.wif_provider.is_some() {
                    // Pure async I/O (assume the AWS broker role, sign and exchange the SigV4
                    // subject token) -- deliberately not routed through spawn_bounded_blocking, so
                    // it is not bounded by BLOCKING_BUILD_PERMITS. Its concurrency is instead
                    // bounded by per-key single-flight (this same cache) and by the number of
                    // distinct registered providers/keys, which is admin-admitted.
                    federation::build_federated_source(
                        &registry.federated_access_token_sources,
                        spec,
                    )
                    .await
                } else {
                    registry.build_credentials(spec).await
                };
                histogram!(GCP_CREDENTIAL_BUILD_DURATION).record(start.elapsed().as_secs_f64());
                result
            })
            .map_err(|_| GcpAuthError::Build {
                audience: audience.clone(),
                message: "TaskCenter is shutting down".to_owned(),
            })?;
        let result = task.await.unwrap_or_else(|_| {
            Err(GcpAuthError::Build {
                audience,
                message: "GCP credential construction task failed".to_owned(),
            })
        });
        counter!(
            GCP_CREDENTIAL_BUILDS,
            "result" => if result.is_ok() { RESULT_SUCCESS } else { RESULT_ERROR }
        )
        .increment(1);
        result
    }

    /// Resolves the credential(s) needed for `spec` and builds the outer ID-token credential,
    /// offloading the blocking `.build()` call. The impersonated arm first resolves the
    /// process-wide ambient source: a permanent failure here can only ever strand the outer
    /// refresh task, since the shared source's refresh task is independent of any single key and
    /// is recovered separately (see [`CredentialRegistry::recover_ambient_source_if_dead`]). Never
    /// called for a federated spec -- see [`CredentialRegistry::build_on_tc_task`].
    async fn build_credentials(
        &self,
        spec: IdTokenSpec,
    ) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
        let IdTokenSpec {
            impersonate,
            audience,
            ..
        } = spec;
        // The blocking closure takes ownership of the strings; one audience copy stays behind as
        // run_blocking's panic context (builder errors already carry the audience themselves).
        let panic_context = audience.clone();
        match impersonate {
            None => run_blocking(panic_context, move || build_ambient_credentials(&audience)).await,
            Some(sa) => {
                let source = self
                    .ambient_source()
                    .await
                    .map_err(|message| GcpAuthError::Adc {
                        audience: audience.clone(),
                        impersonate: sa.clone(),
                        message,
                    })?;
                run_blocking(panic_context, move || {
                    build_impersonated_credentials(&audience, &sa, source)
                })
                .await
            }
        }
    }
}

/// A permanent error probing `source`'s own cached token state proves its refresh task already
/// published that error and exited -- replacing it strands nothing live. Anything else (a token, a
/// transient error, or the probe timing out) means the refresh task might still be alive, so it
/// must be kept.
///
/// Shared by the ambient source (see [`CredentialRegistry::recover_ambient_source_if_dead`]) and,
/// per WIF provider, the federated access-token source (see
/// [`federation::recover_federated_access_token_source_if_dead`]).
async fn credentials_source_is_dead(source: &google_cloud_auth::credentials::Credentials) -> bool {
    matches!(
        tokio::time::timeout(SOURCE_PROBE_TIMEOUT, source.headers(http::Extensions::new())).await,
        Ok(Err(e)) if !e.is_transient()
    )
}

/// Bounds concurrent blocking `google-cloud-auth`/ADC builds. Acquired strictly around the
/// `spawn_blocking` call in [`spawn_bounded_blocking`] -- not around a whole key's construction --
/// so a permit is never held while waiting on another. In particular,
/// [`CredentialRegistry::build_credentials`] resolves the ambient source before entering this
/// bound for the outer credential build.
static BLOCKING_BUILD_PERMITS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_BLOCKING_BUILDS);

async fn spawn_bounded_blocking<T: Send + 'static>(
    build: impl FnOnce() -> T + Send + 'static,
) -> Result<T, tokio::task::JoinError> {
    let _permit = BLOCKING_BUILD_PERMITS
        .acquire()
        .await
        .expect("BLOCKING_BUILD_PERMITS is never closed");
    tokio::task::spawn_blocking(build).await
}

/// Runs `build` on a blocking thread, mapping a panicked build thread to a `GcpAuthError` at this
/// one call site.
async fn run_blocking(
    audience: String,
    build: impl FnOnce() -> Result<Arc<dyn IdTokenSource>, GcpAuthError> + Send + 'static,
) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
    spawn_bounded_blocking(build).await.unwrap_or_else(|e| {
        Err(GcpAuthError::Build {
            audience,
            message: format!("construction thread panicked: {e}"),
        })
    })
}

/// Builds the outer credential for a non-impersonated token.
fn build_ambient_credentials(audience: &str) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
    use google_cloud_auth::credentials::idtoken;

    let credentials = idtoken::Builder::new(audience).build().map_err(|e| {
        // authorized_user (gcloud) and external_account (Workload Identity Federation) ADC
        // sources cannot mint ID tokens directly.
        if e.is_not_supported() {
            GcpAuthError::AmbientUnsupported {
                audience: audience.to_owned(),
            }
        } else {
            GcpAuthError::Build {
                audience: audience.to_owned(),
                message: e.to_string(),
            }
        }
    })?;
    Ok(Arc::new(Live(credentials)) as Arc<dyn IdTokenSource>)
}

/// Builds the outer credential for an impersonated key, from the shared ambient `source`.
fn build_impersonated_credentials(
    audience: &str,
    service_account: &str,
    source: google_cloud_auth::credentials::Credentials,
) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> {
    use google_cloud_auth::credentials::idtoken;

    let credentials =
        idtoken::impersonated::Builder::from_source_credentials(audience, service_account, source)
            .build()
            .map_err(|e| GcpAuthError::Build {
                audience: audience.to_owned(),
                message: e.to_string(),
            })?;
    Ok(Arc::new(Live(credentials)) as Arc<dyn IdTokenSource>)
}

/// Token-mint client: a cheap handle to the process-global credential registry. `ServiceClient`
/// clones share cached credentials and their refresh tasks. Outside tests the only state this
/// handle carries is the `Handle` it was constructed with, which it passes explicitly to
/// [`credential_registry`] on every mint -- `mint()` is reachable from invoker invocation tasks,
/// which have no `TaskCenter` task-local of their own, so this client never relies on one.
#[derive(Clone)]
pub struct GcpTokenClient {
    task_center: Handle,
    #[cfg(any(test, feature = "test_util"))]
    inner: Arc<Inner>,
}

/// Instance-local test overlay. Kept separate from the global registry so tests that construct
/// multiple `ServiceClient`s in one process cannot cross-contaminate each other by seeding a
/// shared cache entry.
#[cfg(any(test, feature = "test_util"))]
struct Inner {
    test_force_failure: Mutex<Option<String>>,
    test_sources: Mutex<HashMap<IdTokenSpec, Arc<dyn IdTokenSource>>>,
}

impl GcpTokenClient {
    /// The caller captures `task_center` at construction; minting never reads a TaskCenter
    /// task-local from the calling task.
    pub fn new(task_center: Handle) -> Self {
        Self {
            task_center,
            #[cfg(any(test, feature = "test_util"))]
            inner: Arc::new(Inner {
                test_force_failure: Mutex::new(None),
                test_sources: Mutex::default(),
            }),
        }
    }

    /// Mint an OIDC ID token for the given audience. If `wif_provider` is set, the token is
    /// minted through the AWS -> GCP workload identity federation chain (see [`federation`]),
    /// impersonating `impersonate_service_account` (required in this case). Otherwise, if
    /// `impersonate_service_account` is set, the token is minted via the IAM Credentials
    /// `generateIdToken` API for that service account from ambient ADC identity; with neither
    /// set, it is minted from ambient ADC identity directly.
    pub async fn mint(
        &self,
        wif_provider: Option<&str>,
        impersonate_service_account: Option<&str>,
        audience: &str,
    ) -> Result<String, GcpAuthError> {
        let spec = IdTokenSpec {
            wif_provider: wif_provider.map(str::to_owned),
            impersonate: impersonate_service_account.map(str::to_owned),
            audience: audience.to_owned(),
        };
        let impersonate = impersonate_service_account
            .unwrap_or("(ambient)")
            .to_owned();
        let mode = if wif_provider.is_some() {
            MINT_MODE_FEDERATED
        } else {
            MINT_MODE_ADC
        };

        // Test seeds/forced failures are never evicted: they live on this instance, not in the
        // shared registry cache. `registry` is resolved at most once per mint call and reused for
        // both the build and, on a permanent failure, eviction/recovery below -- those must act on
        // the same registry generation that built the source, not whichever one happens to occupy
        // the global slot by the time the error is handled.
        let (source, registry) = match self.test_intercept(&spec, &impersonate) {
            Some(Ok(source)) => (source, None),
            Some(Err(error)) => return Err(error),
            None => {
                let registry = match credential_registry(&self.task_center) {
                    Ok(registry) => registry,
                    Err(message) => {
                        counter!(GCP_TOKEN_MINTS, "outcome" => MINT_OUTCOME_BUILD_ERROR, "mode" => mode)
                            .increment(1);
                        return Err(GcpAuthError::Build {
                            audience: audience.to_owned(),
                            message,
                        });
                    }
                };
                match registry.get_or_build(&spec).await {
                    Ok(source) => (source, Some(registry)),
                    Err(error) => {
                        // Each failed caller counts: a single failed single-flight build fails
                        // every caller waiting on it, not just one.
                        counter!(GCP_TOKEN_MINTS, "outcome" => MINT_OUTCOME_BUILD_ERROR, "mode" => mode)
                            .increment(1);
                        return Err(error);
                    }
                }
            }
        };

        match tokio::time::timeout(MINT_ATTEMPT_TIMEOUT, source.id_token()).await {
            Ok(Ok(token)) => {
                counter!(GCP_TOKEN_MINTS, "outcome" => MINT_OUTCOME_SUCCESS, "mode" => mode)
                    .increment(1);
                Ok(token)
            }
            Ok(Err(error)) => {
                counter!(
                    GCP_TOKEN_MINTS,
                    "outcome" => if error.is_transient() {
                        MINT_OUTCOME_TRANSIENT_ERROR
                    } else {
                        MINT_OUTCOME_PERMANENT_ERROR
                    },
                    "mode" => mode
                )
                .increment(1);
                // Transient failures self-heal via the credential's own refresh loop, no need to
                // evict the entry; a permanent failure will not recover, so evict it -- but only if
                // the cache still holds the exact credential that produced the error.
                if let Some(registry) = &registry
                    && !error.is_transient()
                {
                    registry.evict_if_unchanged(&spec, &source).await;
                    // A federated key shares its source with every other key using the same WIF
                    // provider, not with the ambient identity, so the two recovery paths are
                    // mutually exclusive.
                    if let Some(provider) = &spec.wif_provider {
                        federation::recover_federated_access_token_source_if_dead(
                            &registry.federated_access_token_sources,
                            provider,
                        )
                        .await;
                    } else if spec.impersonate.is_some() {
                        registry.recover_ambient_source_if_dead().await;
                    }
                }
                Err(GcpAuthError::Mint {
                    audience: audience.to_owned(),
                    impersonate,
                    message: display_error_chain(&error),
                })
            }
            Err(_) => {
                counter!(GCP_TOKEN_MINTS, "outcome" => MINT_OUTCOME_TIMEOUT, "mode" => mode)
                    .increment(1);
                Err(GcpAuthError::Timeout {
                    audience: audience.to_owned(),
                    impersonate,
                    duration: MINT_ATTEMPT_TIMEOUT,
                })
            }
        }
    }

    /// Test seam consulted at the top of `mint()`: a forced failure or a seeded source, if any.
    /// One expression in `mint()` covers both builds, so production and test compilation cannot
    /// drift apart.
    #[cfg(any(test, feature = "test_util"))]
    fn test_intercept(
        &self,
        spec: &IdTokenSpec,
        impersonate: &str,
    ) -> Option<Result<Arc<dyn IdTokenSource>, GcpAuthError>> {
        if let Some(message) = self.inner.test_force_failure.lock().clone() {
            return Some(Err(GcpAuthError::Mint {
                audience: spec.audience.clone(),
                impersonate: impersonate.to_owned(),
                message,
            }));
        }
        self.inner.test_sources.lock().get(spec).cloned().map(Ok)
    }

    #[cfg(not(any(test, feature = "test_util")))]
    #[inline(always)]
    fn test_intercept(
        &self,
        _spec: &IdTokenSpec,
        _impersonate: &str,
    ) -> Option<Result<Arc<dyn IdTokenSource>, GcpAuthError>> {
        None
    }

    #[cfg(any(test, feature = "test_util"))]
    pub fn force_mint_failure_for_test(&self, message: &str) {
        *self.inner.test_force_failure.lock() = Some(message.to_owned());
    }

    /// Test-only: seed a token so subsequent `mint` calls with the same key return it directly.
    /// Seeding is local to the `GcpTokenClient` instance, not the shared registry, so tests that
    /// construct multiple `ServiceClient`s in one process do not interfere with each other.
    #[cfg(any(test, feature = "test_util"))]
    pub fn seed_for_test(&self, impersonate: Option<&str>, audience: &str, token: String) {
        struct MockToken(String);

        #[async_trait]
        impl IdTokenSource for MockToken {
            async fn id_token(
                &self,
            ) -> Result<String, google_cloud_auth::errors::CredentialsError> {
                Ok(self.0.clone())
            }
        }

        let spec = IdTokenSpec {
            wif_provider: None,
            impersonate: impersonate.map(str::to_owned),
            audience: audience.to_owned(),
        };
        self.inner
            .test_sources
            .lock()
            .insert(spec, Arc::new(MockToken(token)));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use restate_core::TaskCenter;

    use super::*;

    fn token() -> String {
        "test-token".to_owned()
    }

    #[test]
    fn ambient_unsupported_error_is_actionable_and_leak_free() {
        let err = GcpAuthError::AmbientUnsupported {
            audience: "https://svc-abc-uc.a.run.app".into(),
        };
        let msg = err.to_string();
        // Actionable: names the audience and the fix.
        assert!(msg.contains("https://svc-abc-uc.a.run.app"), "{msg}");
        assert!(msg.contains("--gcp-impersonate-service-account"), "{msg}");
        // Leak-free: must not surface the google-cloud-auth internal API hint.
        assert!(!msg.contains("idtoken::user_account"), "{msg}");
        assert!(!msg.to_lowercase().contains("builder directly"), "{msg}");
    }

    /// The single most common federation/impersonation misconfiguration -- a customer forgetting
    /// to grant `roles/iam.serviceAccountOpenIdTokenCreator` on the invocation service account --
    /// surfaces from `iamcredentials.googleapis.com` as a 403 with a `PERMISSION_DENIED` status in
    /// the JSON body. `CredentialsError::Display` alone drops this (it only prints its own
    /// top-level message); `display_error_chain` must recover it from the source chain so it
    /// reaches `GcpAuthError::Mint`'s message, legible in `restate dp register` errors and server
    /// logs.
    #[test]
    fn credentials_error_403_permission_denied_survives_display_error_chain() {
        let body = br#"{"error":{"code":403,"message":"The caller does not have permission","status":"PERMISSION_DENIED"}}"#;
        let gax_error = google_cloud_gax::error::Error::http(
            403,
            http::HeaderMap::new(),
            bytes::Bytes::from_static(body),
        );
        let credentials_error = google_cloud_auth::errors::CredentialsError::new(
            false,
            "failed to fetch ID token via impersonation",
            gax_error,
        );

        let message = display_error_chain(&credentials_error);
        assert!(message.contains("403"), "{message}");
        assert!(message.contains("PERMISSION_DENIED"), "{message}");
    }

    struct MockSource {
        calls: AtomicUsize,
        behavior: Mutex<Box<dyn FnMut(usize) -> MockOutcome + Send>>,
    }

    enum MockOutcome {
        Token(String),
        Error(google_cloud_auth::errors::CredentialsError),
    }

    impl MockSource {
        fn new(behavior: impl FnMut(usize) -> MockOutcome + Send + 'static) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                behavior: Mutex::new(Box::new(behavior)),
            })
        }
    }

    #[async_trait]
    impl IdTokenSource for MockSource {
        async fn id_token(&self) -> Result<String, google_cloud_auth::errors::CredentialsError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let outcome = (self.behavior.lock())(call);
            match outcome {
                MockOutcome::Token(token) => Ok(token),
                MockOutcome::Error(error) => Err(error),
            }
        }
    }

    impl IdTokenSpec {
        fn ambient(audience: &str) -> Self {
            IdTokenSpec {
                wif_provider: None,
                impersonate: None,
                audience: audience.to_owned(),
            }
        }

        fn impersonated(audience: &str, service_account: &str) -> Self {
            IdTokenSpec {
                wif_provider: None,
                impersonate: Some(service_account.to_owned()),
                audience: audience.to_owned(),
            }
        }

        fn federated(audience: &str, provider: &str, service_account: &str) -> Self {
            IdTokenSpec {
                wif_provider: Some(provider.to_owned()),
                impersonate: Some(service_account.to_owned()),
                audience: audience.to_owned(),
            }
        }
    }

    fn transient_error(message: &str) -> google_cloud_auth::errors::CredentialsError {
        google_cloud_auth::errors::CredentialsError::from_msg(true, message)
    }

    fn permanent_error(message: &str) -> google_cloud_auth::errors::CredentialsError {
        google_cloud_auth::errors::CredentialsError::from_msg(false, message)
    }

    /// Test convenience wrapping [`credential_registry`] with the ambient `TaskCenter::current()`
    /// handle: every test in this module runs either inside `#[restate_core::test]` or inside a
    /// future wrapped with `.in_tc(&tc)`, both of which set that task-local, so this always
    /// resolves the same registry a same-scoped `GcpTokenClient::new(TaskCenter::current())`
    /// would.
    fn credential_registry_for_test() -> Arc<CredentialRegistry> {
        credential_registry(&TaskCenter::current()).expect("task center is not shutting down")
    }

    fn add_build_override(
        cache_key: IdTokenSpec,
        f: impl Fn(&IdTokenSpec) -> Result<Arc<dyn IdTokenSource>, GcpAuthError> + Send + Sync + 'static,
    ) {
        credential_registry_for_test()
            .test_hooks
            .build_overrides
            .lock()
            .insert(cache_key, Arc::new(f));
    }

    fn add_ambient_source_override(
        f: impl Fn() -> Result<google_cloud_auth::credentials::Credentials, String>
        + Send
        + Sync
        + 'static,
    ) {
        *credential_registry_for_test()
            .test_hooks
            .ambient_source_override
            .lock() = Some(Arc::new(f));
    }

    /// A `CredentialsProvider` a test can drive deterministically, with no SDK dependencies:
    /// `Credentials::from(...)` wraps this so its `headers()` impl below is exactly what
    /// `credentials_source_is_dead` observes.
    #[derive(Clone, Copy, Debug)]
    enum ProbeOutcome {
        Healthy,
        Dead,
        Transient,
        Hang,
    }

    struct FakeCredentialsProvider(Mutex<Box<dyn FnMut() -> ProbeOutcome + Send>>);

    impl std::fmt::Debug for FakeCredentialsProvider {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("FakeCredentialsProvider")
        }
    }

    impl FakeCredentialsProvider {
        fn always(outcome: impl Fn() -> ProbeOutcome + Send + 'static) -> Self {
            Self(Mutex::new(Box::new(outcome)))
        }
    }

    impl google_cloud_auth::credentials::CredentialsProvider for FakeCredentialsProvider {
        async fn headers(
            &self,
            _extensions: http::Extensions,
        ) -> std::result::Result<
            google_cloud_auth::credentials::CacheableResource<http::HeaderMap>,
            google_cloud_auth::errors::CredentialsError,
        > {
            let outcome = (self.0.lock())();
            match outcome {
                ProbeOutcome::Healthy => {
                    Ok(google_cloud_auth::credentials::CacheableResource::New {
                        entity_tag: google_cloud_auth::credentials::EntityTag::new(),
                        data: http::HeaderMap::new(),
                    })
                }
                ProbeOutcome::Dead => Err(permanent_error("source refresh task permanently dead")),
                ProbeOutcome::Transient => Err(transient_error(
                    "source refresh task transiently unavailable",
                )),
                ProbeOutcome::Hang => std::future::pending().await,
            }
        }

        async fn universe_domain(&self) -> Option<String> {
            None
        }
    }

    #[restate_core::test]
    async fn single_flight_builds_once_under_concurrent_misses() {
        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://single-flight.example.com";
        let builds = Arc::new(AtomicUsize::new(0));

        add_build_override(IdTokenSpec::ambient(audience), {
            let builds = builds.clone();
            move |_| {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            }
        });

        let results =
            futures::future::join_all((0..64).map(|_| client.mint(None, None, audience))).await;

        assert!(results.iter().all(|r| r.is_ok()), "{results:?}");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    /// Pins P1's fix (restatedev/restate#5151): the impersonated arm's source ADC credential is a
    /// single process-wide refresh task, not one per key. Proving `ambient_source` single-flights
    /// and shares its result is equivalent to proving N concurrent impersonated constructions
    /// share one source build.
    ///
    /// Relies on `ambient_source` being uninitialized when this test starts: `#[restate_core::test]`
    /// builds a fresh `TaskCenter` per test, and `credential_registry` builds a fresh
    /// `CredentialRegistry` for each new `TaskCenter` handle it sees (see that function's doc
    /// comment), so this test gets its own registry regardless of what ran before it in the same
    /// process.
    #[restate_core::test]
    async fn impersonated_constructions_share_one_ambient_source_build() {
        let build_count = Arc::new(AtomicUsize::new(0));
        add_ambient_source_override({
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let registry = credential_registry_for_test();
        let results = futures::future::join_all((0..8).map(|_| registry.ambient_source())).await;

        assert!(results.iter().all(|r| r.is_ok()), "{results:?}");
        assert_eq!(build_count.load(Ordering::SeqCst), 1);
    }

    #[restate_core::test]
    async fn credentials_source_is_dead_only_for_a_proven_permanent_error() {
        let cases = [
            (ProbeOutcome::Healthy, false),
            (ProbeOutcome::Transient, false),
            (ProbeOutcome::Dead, true),
            (ProbeOutcome::Hang, false),
        ];
        for (outcome, expected_dead) in cases {
            let source = google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(move || outcome),
            );
            assert_eq!(
                credentials_source_is_dead(&source).await,
                expected_dead,
                "{outcome:?}"
            );
        }
    }

    /// A shared ambient source whose refresh task has permanently died is replaced exactly once,
    /// by the first permanent impersonated mint failure to probe it -- and the replacement is then
    /// reused without further rebuilds.
    #[restate_core::test]
    async fn dead_ambient_source_is_replaced_after_permanent_impersonation_failure() {
        credential_registry_for_test()
            .ambient_source
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Dead),
            ))
            .await;

        let build_count = Arc::new(AtomicUsize::new(0));
        add_ambient_source_override({
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://ambient-recovery.example.com";
        let service_account = "sa@example.iam.gserviceaccount.com";
        add_build_override(IdTokenSpec::impersonated(audience, service_account), |_| {
            Ok(MockSource::new(|_| {
                MockOutcome::Error(permanent_error("impersonation misconfigured"))
            }) as Arc<dyn IdTokenSource>)
        });

        let outcome = client.mint(None, Some(service_account), audience).await;
        assert!(
            matches!(outcome, Err(GcpAuthError::Mint { .. })),
            "{outcome:?}"
        );

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            1,
            "the dead source must be replaced exactly once"
        );

        // The replacement is healthy and reusable without a further rebuild.
        assert!(
            credential_registry_for_test()
                .ambient_source()
                .await
                .is_ok()
        );
        assert_eq!(build_count.load(Ordering::SeqCst), 1);
    }

    /// A healthy shared ambient source is never replaced by a repeatedly-failing impersonation
    /// target: the failure is scoped to that one key, and the source is provably fine.
    #[restate_core::test]
    async fn healthy_ambient_source_is_not_replaced_by_repeated_impersonation_failures() {
        credential_registry_for_test()
            .ambient_source
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
            ))
            .await;

        let build_count = Arc::new(AtomicUsize::new(0));
        add_ambient_source_override({
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://ambient-stable.example.com";
        let service_account = "sa@example.iam.gserviceaccount.com";
        add_build_override(IdTokenSpec::impersonated(audience, service_account), |_| {
            Ok(MockSource::new(|_| {
                MockOutcome::Error(permanent_error("impersonation misconfigured"))
            }) as Arc<dyn IdTokenSource>)
        });

        for _ in 0..5 {
            let outcome = client.mint(None, Some(service_account), audience).await;
            assert!(
                matches!(outcome, Err(GcpAuthError::Mint { .. })),
                "{outcome:?}"
            );
        }

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            0,
            "a healthy source must never be replaced by an impersonation-only failure"
        );
    }

    #[restate_core::test]
    async fn transient_error_keeps_entry_and_self_heals() {
        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://transient.example.com";
        let cache_key = IdTokenSpec::ambient(audience);
        let source = MockSource::new(|call| {
            if call == 0 {
                MockOutcome::Error(transient_error("temporarily unavailable"))
            } else {
                MockOutcome::Token(token())
            }
        });
        let dyn_source: Arc<dyn IdTokenSource> = source.clone();
        credential_registry_for_test()
            .cache
            .insert(cache_key.clone(), dyn_source.clone())
            .await;

        let first = client.mint(None, None, audience).await;
        assert!(matches!(first, Err(GcpAuthError::Mint { .. })), "{first:?}");

        // The entry must still be present and unchanged (no eviction on transient failure).
        let still_cached = credential_registry_for_test().cache.get(&cache_key).await;
        assert!(matches!(still_cached, Some(s) if Arc::ptr_eq(&s, &dyn_source)));

        // The mock "self-heals" on the next call, as a real credential's refresh loop would.
        let second = client.mint(None, None, audience).await;
        assert!(second.is_ok(), "{second:?}");
    }

    #[restate_core::test]
    async fn permanent_error_evicts_conditionally() {
        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://permanent.example.com";
        let cache_key = IdTokenSpec::ambient(audience);
        let source: Arc<dyn IdTokenSource> =
            MockSource::new(|_| MockOutcome::Error(permanent_error("misconfigured")));
        credential_registry_for_test()
            .cache
            .insert(cache_key.clone(), source.clone())
            .await;

        let outcome = client.mint(None, None, audience).await;
        assert!(
            matches!(outcome, Err(GcpAuthError::Mint { .. })),
            "{outcome:?}"
        );

        assert!(
            credential_registry_for_test()
                .cache
                .get(&cache_key)
                .await
                .is_none()
        );
    }

    #[restate_core::test]
    async fn aba_race_stale_caller_evict_is_a_no_op() {
        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://aba.example.com";
        let cache_key = IdTokenSpec::ambient(audience);
        let new_source: Arc<dyn IdTokenSource> = MockSource::new(|_| MockOutcome::Token(token()));

        /// Simulates a concurrent rebuild completing -- replacing this source in the registry
        /// cache with `replacement` -- before this permanently-failing source's own error is
        /// reported back to `mint()`. Driving the swap from inside `id_token()` exercises the
        /// exact race `evict_if_unchanged`'s compare-and-evict guards against, through the real
        /// `mint()` call path rather than by hand-simulating the two steps independently.
        struct SwapThenFail {
            spec: IdTokenSpec,
            replacement: Arc<dyn IdTokenSource>,
        }

        #[async_trait]
        impl IdTokenSource for SwapThenFail {
            async fn id_token(
                &self,
            ) -> Result<String, google_cloud_auth::errors::CredentialsError> {
                credential_registry_for_test()
                    .cache
                    .insert(self.spec.clone(), self.replacement.clone())
                    .await;
                Err(permanent_error("old, now gone"))
            }
        }

        let old_source: Arc<dyn IdTokenSource> = Arc::new(SwapThenFail {
            spec: cache_key.clone(),
            replacement: new_source.clone(),
        });
        credential_registry_for_test()
            .cache
            .insert(cache_key.clone(), old_source.clone())
            .await;

        let outcome = client.mint(None, None, audience).await;
        assert!(
            matches!(outcome, Err(GcpAuthError::Mint { .. })),
            "{outcome:?}"
        );

        let cached = credential_registry_for_test().cache.get(&cache_key).await;
        assert!(
            matches!(cached, Some(s) if Arc::ptr_eq(&s, &new_source)),
            "evict from a stale caller must not remove the freshly rebuilt healthy entry"
        );
    }

    /// N cold outer constructions for the same WIF provider share exactly one build of that
    /// provider's shared access-token source credentials -- the federated analogue of
    /// `impersonated_constructions_share_one_ambient_source_build`. Exercised through `mint()` on
    /// N distinct audiences (N distinct outer `IdTokenSpec`s, so N distinct outer credentials)
    /// rather than through the source-sharing primitive directly, since the point is to prove
    /// sharing holds through the real per-key construction path.
    #[restate_core::test]
    async fn n_federated_keys_for_one_provider_share_one_source_build() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/share";
        let build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let client = GcpTokenClient::new(TaskCenter::current());
        let service_account = "sa@example.iam.gserviceaccount.com";
        let results = futures::future::join_all((0..8).map(|i| {
            let client = client.clone();
            let audience = format!("https://federated-source-share-{i}.example.com");
            async move {
                client
                    .mint(Some(provider), Some(service_account), &audience)
                    .await
            }
        }))
        .await;

        // Each key's outer construction is real (not overridden), so the resulting mint may well
        // fail (no real network in this sandbox reaches Google); only the source-sharing count
        // below is under test.
        let _ = results;
        assert_eq!(
            build_count.load(Ordering::SeqCst),
            1,
            "8 keys for one WIF provider must share exactly one access-token source build"
        );
    }

    /// A live outer credential's leased access-token source must be reused indefinitely, no matter
    /// how much idle time passes, as long as something keeps touching (and so keeps alive) the
    /// outer key that leases it -- fixing an earlier version of this test, which advanced virtual
    /// time without ever touching the first outer key again. Under the current, reference-driven
    /// model that shape is invalid: the outer *moka* cache itself has a time-to-idle, so an
    /// untouched outer key would legitimately expire and drop its lease well before the span this
    /// test advanced through, which would make a second source build the CORRECT outcome, not a
    /// bug. Reads the cached entry directly (rather than through `mint()`) to touch it: a real
    /// federated outer credential's own `id_token()` does real network I/O and, in a sandboxed
    /// test environment with no route to Google, fails -- and if that failure classifies as
    /// permanent (plausible for a connection error), `mint()` would itself evict the entry via
    /// `evict_if_unchanged`, defeating the very thing this test needs to hold the key alive.
    /// Reading the entry sidesteps that non-determinism while still exercising moka's real
    /// idle-timer reset on access.
    #[restate_core::test(start_paused = true)]
    async fn federated_access_token_source_is_reused_while_its_outer_credential_stays_referenced() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/no-evict-test";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let registry = credential_registry_for_test();
        let first_spec = IdTokenSpec::federated(
            "https://no-evict-first.example.com",
            provider,
            service_account,
        );
        let outer = federation::build_federated_source(
            &registry.federated_access_token_sources,
            first_spec.clone(),
        )
        .await
        .expect("outer construction succeeds");
        registry.cache.insert(first_spec.clone(), outer).await;
        assert_eq!(build_count.load(Ordering::SeqCst), 1);

        // Each hop is well under CACHE_TIME_TO_IDLE (1h), so the outer moka entry for
        // `first_spec` never actually idles out; four hops cover a span (2h) far exceeding the
        // old, invalid test's single un-touched advance.
        for _ in 0..4 {
            assert!(registry.cache.get(&first_spec).await.is_some());
            tokio::time::advance(Duration::from_secs(1800)).await;
        }

        let second_spec = IdTokenSpec::federated(
            "https://no-evict-second.example.com",
            provider,
            service_account,
        );
        federation::build_federated_source(&registry.federated_access_token_sources, second_spec)
            .await
            .expect("outer construction succeeds");
        assert_eq!(
            build_count.load(Ordering::SeqCst),
            1,
            "a live outer credential's access-token source must be reused by a second outer \
             credential for the same provider, never rebuilt"
        );
    }

    /// Two distinct outer credentials for the same provider must each hold their own strong lease
    /// on the exact same access-token source -- not merely resolve to the same pointer, but keep
    /// it alive by refcount. Captures the map's `Weak` entry directly (never
    /// `federated_access_token_source`, which would hand back an extra strong `Arc` and inflate
    /// the count under test) and follows `strong_count()` through 2 -> 1 -> 0 as each cached outer
    /// credential is evicted in turn, proving the leases -- not this test -- are what keep it
    /// alive.
    #[restate_core::test]
    async fn two_outer_credentials_for_one_provider_share_the_same_access_token_source() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/shared-lease";
        let service_account = "sa@example.iam.gserviceaccount.com";
        federation::install_federated_access_token_source_override_for_test(provider, || {
            Ok(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
            ))
        });

        let registry = credential_registry_for_test();
        let spec_a = IdTokenSpec::federated(
            "https://shared-lease-a.example.com",
            provider,
            service_account,
        );
        let spec_b = IdTokenSpec::federated(
            "https://shared-lease-b.example.com",
            provider,
            service_account,
        );
        let outer_a = federation::build_federated_source(
            &registry.federated_access_token_sources,
            spec_a.clone(),
        )
        .await
        .expect("outer construction succeeds");
        let outer_b = federation::build_federated_source(
            &registry.federated_access_token_sources,
            spec_b.clone(),
        )
        .await
        .expect("outer construction succeeds");

        let weak = registry
            .federated_access_token_sources
            .lock()
            .get(provider)
            .cloned()
            .expect("the builds above must have created an entry for this provider");

        registry.cache.insert(spec_a.clone(), outer_a).await;
        registry.cache.insert(spec_b.clone(), outer_b).await;
        assert_eq!(
            weak.strong_count(),
            2,
            "both cached outer credentials must hold their own lease on the shared source"
        );

        registry.cache.invalidate(&spec_a).await;
        registry.cache.run_pending_tasks().await;
        assert_eq!(
            weak.strong_count(),
            1,
            "evicting one outer credential must drop exactly its own lease"
        );

        registry.cache.invalidate(&spec_b).await;
        registry.cache.run_pending_tasks().await;
        assert_eq!(
            weak.strong_count(),
            0,
            "evicting the last outer credential must drop the last lease"
        );
    }

    /// Expiring one outer credential must not reap a still-shared access-token source while
    /// another outer credential for the same provider keeps it alive. Moves the second outer
    /// credential straight into the cache without retaining a local strong reference, so the
    /// post-eviction check -- upgrading the map's own `Weak` -- can only succeed because the
    /// CACHE itself, not this test, is holding the lease.
    #[restate_core::test]
    async fn expiring_one_outer_credential_does_not_reap_a_still_referenced_source() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/still-referenced";
        let service_account = "sa@example.iam.gserviceaccount.com";
        federation::install_federated_access_token_source_override_for_test(provider, || {
            Ok(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
            ))
        });

        let registry = credential_registry_for_test();
        let spec_a =
            IdTokenSpec::federated("https://still-ref-a.example.com", provider, service_account);
        let spec_b =
            IdTokenSpec::federated("https://still-ref-b.example.com", provider, service_account);
        let outer_a = federation::build_federated_source(
            &registry.federated_access_token_sources,
            spec_a.clone(),
        )
        .await
        .expect("outer construction succeeds");
        let outer_b = federation::build_federated_source(
            &registry.federated_access_token_sources,
            spec_b.clone(),
        )
        .await
        .expect("outer construction succeeds");
        registry.cache.insert(spec_a.clone(), outer_a).await;
        registry.cache.insert(spec_b, outer_b).await;

        // Expire and evict only spec_a's outer entry; spec_b's stays cached, and only the CACHE
        // still references the shared source from here on.
        registry.cache.invalidate(&spec_a).await;
        registry.cache.run_pending_tasks().await;

        let live = federation::reap_disused_federated_access_token_sources(
            &registry.federated_access_token_sources,
        );
        assert_eq!(
            live, 1,
            "the shared access-token source must still be counted live: spec_b's outer \
             credential still references it"
        );
        assert!(
            registry
                .federated_access_token_sources
                .lock()
                .get(provider)
                .and_then(Weak::upgrade)
                .is_some(),
            "the shared access-token source must not be reaped while spec_b's outer credential \
             still references it"
        );
    }

    /// After the FINAL outer credential referencing a provider expires and is evicted, the
    /// housekeeping reap must remove that provider's access-token source entirely -- the weak
    /// entry no longer upgrades, the map key itself is gone (not merely a dead tombstone), and the
    /// live count fed into `gcp.federation.sources.active` reflects the removal. A later mint for
    /// the same provider must then build a fresh access-token source, not reuse anything.
    #[restate_core::test]
    async fn reap_removes_a_federated_access_token_source_after_its_last_outer_credential_expires()
    {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/reap-test";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let registry = credential_registry_for_test();
        let spec =
            IdTokenSpec::federated("https://reap-test.example.com", provider, service_account);
        let outer = federation::build_federated_source(
            &registry.federated_access_token_sources,
            spec.clone(),
        )
        .await
        .expect("outer construction succeeds");
        registry.cache.insert(spec.clone(), outer).await;
        assert_eq!(build_count.load(Ordering::SeqCst), 1);

        // The cache is now the sole strong referent. Evict it -- the last lease -- and reap.
        registry.cache.invalidate(&spec).await;
        registry.cache.run_pending_tasks().await;
        let live = federation::reap_disused_federated_access_token_sources(
            &registry.federated_access_token_sources,
        );
        assert_eq!(
            live, 0,
            "the live count must reflect removal once the last outer credential expires"
        );
        assert!(
            registry
                .federated_access_token_sources
                .lock()
                .get(provider)
                .is_none(),
            "the map key itself must be gone after reaping, not merely a dead tombstone"
        );

        // A later mint for the same provider must build a fresh source, not resurrect anything.
        let second_spec = IdTokenSpec::federated(
            "https://reap-test-second.example.com",
            provider,
            service_account,
        );
        federation::build_federated_source(&registry.federated_access_token_sources, second_spec)
            .await
            .expect("outer construction succeeds");
        assert_eq!(
            build_count.load(Ordering::SeqCst),
            2,
            "a provider whose access-token source was fully reaped must build exactly one fresh \
             source on its next reference"
        );
    }

    /// A failed access-token build must not permanently retain anything: the dead weak tombstone
    /// it leaves behind is both prunable by housekeeping and independently replaceable by the very
    /// next lookup, whichever comes first.
    #[restate_core::test]
    async fn failed_construction_leaves_no_permanently_retained_source() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/construction-fails";
        let service_account = "sa@example.iam.gserviceaccount.com";
        federation::install_federated_access_token_source_override_for_test(provider, || {
            Err("simulated STS exchange failure".to_owned())
        });

        let registry = credential_registry_for_test();
        let spec = IdTokenSpec::federated(
            "https://construction-fails.example.com",
            provider,
            service_account,
        );
        let result =
            federation::build_federated_source(&registry.federated_access_token_sources, spec)
                .await;
        assert!(
            matches!(&result, Err(GcpAuthError::Adc { .. })),
            "a failed access-token build must fail outer construction as Adc, got is_ok={}",
            result.is_ok()
        );

        assert_eq!(
            federation::reap_disused_federated_access_token_sources(
                &registry.federated_access_token_sources
            ),
            0,
            "a failed build's dead weak entry must be pruned by housekeeping, not retained"
        );

        let build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });
        let retry_spec = IdTokenSpec::federated(
            "https://construction-fails-retry.example.com",
            provider,
            service_account,
        );
        federation::build_federated_source(&registry.federated_access_token_sources, retry_spec)
            .await
            .expect("a fresh lookup must be able to replace the dead tombstone directly");
        assert_eq!(build_count.load(Ordering::SeqCst), 1);
    }

    /// Test-only stand-in for `federation::FederatedIdTokenCredentials`: holds the same kind of
    /// `Arc<FederatedAccessTokenSource>` lease, but lets the test control both the `id_token()`
    /// outcome and inject a side effect at the exact moment `id_token()` runs. Used below to
    /// deterministically simulate housekeeping's reap running in the narrow window between an
    /// outer cache entry being evicted and `recover_federated_access_token_source_if_dead`'s own
    /// upgrade check, without sleeps or real thread interleaving -- the same technique
    /// `SwapThenFail` above uses for the ABA race.
    struct LeasedMockSource {
        spec: IdTokenSpec,
        _access_token_source: Arc<federation::FederatedAccessTokenSource>,
    }

    #[async_trait]
    impl IdTokenSource for LeasedMockSource {
        async fn id_token(&self) -> Result<String, google_cloud_auth::errors::CredentialsError> {
            let registry = credential_registry_for_test();
            // Simulate the outer cache entry for this key already being gone -- as the real
            // `evict_if_unchanged` a moment from now will make it -- with housekeeping's reap
            // squeezed into that exact window, all before this permanent error is even reported
            // back to `mint()`, let alone before recovery starts.
            registry.cache.invalidate(&self.spec).await;
            registry.cache.run_pending_tasks().await;
            let provider = self
                .spec
                .wif_provider
                .as_deref()
                .expect("LeasedMockSource is only used for federated specs");
            federation::reap_disused_federated_access_token_sources(
                &registry.federated_access_token_sources,
            );
            assert!(
                registry
                    .federated_access_token_sources
                    .lock()
                    .get(provider)
                    .and_then(Weak::upgrade)
                    .is_some(),
                "the access-token source must survive a reap that runs while this outer \
                 credential -- still executing its own id_token() call -- keeps it leased"
            );
            Err(permanent_error("impersonation misconfigured"))
        }
    }

    /// Sharpens the recovery/reap race invariant: during permanent-error recovery, `mint()` holds
    /// the failing outer credential locally for the duration of the call, so its lease keeps the
    /// access-token source upgradable even if housekeeping's reap runs in the exact window between
    /// the outer cache entry being evicted and recovery's own upgrade check. Seeds the source as
    /// already dead beforehand, so a reap that wrongly found it absent (rather than merely dead)
    /// would silently skip recovery instead of replacing it -- `recovery_build_count` catches that.
    #[restate_core::test]
    async fn reap_during_the_recovery_window_does_not_defeat_recovery() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/recovery-vs-housekeeping";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let registry = credential_registry_for_test();

        let access_token_source = federation::federated_access_token_source(
            &registry.federated_access_token_sources,
            provider,
        );
        access_token_source
            .credentials
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Dead),
            ))
            .await;

        let recovery_build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let recovery_build_count = recovery_build_count.clone();
            move || {
                recovery_build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let audience = "https://reap-vs-recovery.example.com";
        let spec = IdTokenSpec::federated(audience, provider, service_account);
        add_build_override(spec.clone(), {
            let access_token_source = access_token_source.clone();
            move |spec| {
                Ok(Arc::new(LeasedMockSource {
                    spec: spec.clone(),
                    _access_token_source: access_token_source.clone(),
                }) as Arc<dyn IdTokenSource>)
            }
        });
        // Drop this test's own handle: from here on the only strong reference is the one
        // `LeasedMockSource` (and, through it, mint()'s own local outer credential) holds.
        drop(access_token_source);

        let client = GcpTokenClient::new(TaskCenter::current());
        let outcome = client
            .mint(Some(provider), Some(service_account), audience)
            .await;
        assert!(
            matches!(outcome, Err(GcpAuthError::Mint { .. })),
            "{outcome:?}"
        );

        assert_eq!(
            recovery_build_count.load(Ordering::SeqCst),
            1,
            "recovery must actually replace the dead source, not be defeated by a reap that ran \
             during the recovery window"
        );
    }

    /// A shared federated source whose refresh task has permanently died is replaced exactly once
    /// by the first permanent mint failure on a key targeting that provider to probe it -- the
    /// federated analogue of `dead_ambient_source_is_replaced_after_permanent_impersonation_failure`.
    ///
    /// Drives the outer credential through `LeasedMockSource`, not a plain `MockSource`: a
    /// seeded source with no real lease is already a dead tombstone by the time recovery runs
    /// (see [`federation::FederatedAccessTokenSources`]'s doc for why), so recovery would
    /// correctly no-op instead of replacing it -- not a bug, but a test that needs an actual
    /// lease to mean anything.
    #[restate_core::test]
    async fn dead_federated_source_is_replaced_after_permanent_mint_failure() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/recovery";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let registry = credential_registry_for_test();
        let access_token_source = federation::federated_access_token_source(
            &registry.federated_access_token_sources,
            provider,
        );
        access_token_source
            .credentials
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Dead),
            ))
            .await;

        let build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let audience = "https://federated-recovery.example.com";
        let spec = IdTokenSpec::federated(audience, provider, service_account);
        add_build_override(spec.clone(), {
            let access_token_source = access_token_source.clone();
            move |spec| {
                Ok(Arc::new(LeasedMockSource {
                    spec: spec.clone(),
                    _access_token_source: access_token_source.clone(),
                }) as Arc<dyn IdTokenSource>)
            }
        });
        drop(access_token_source);

        let client = GcpTokenClient::new(TaskCenter::current());
        let outcome = client
            .mint(Some(provider), Some(service_account), audience)
            .await;
        assert!(
            matches!(outcome, Err(GcpAuthError::Mint { .. })),
            "{outcome:?}"
        );

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            1,
            "the dead federated source must be replaced exactly once"
        );

        // The replacement is healthy and reusable without a further rebuild: this build must
        // never run.
        let reused: Result<google_cloud_auth::credentials::Credentials, String> =
            federation::federated_access_token_source(
                &registry.federated_access_token_sources,
                provider,
            )
            .credentials
            .get_or_build(async { unreachable!("the slot must already hold the recovered source") })
            .await;
        assert!(reused.is_ok());
        assert_eq!(build_count.load(Ordering::SeqCst), 1);
    }

    /// A healthy shared federated source is never replaced by a repeatedly-failing impersonation
    /// target on a key using it -- the federated analogue of
    /// `healthy_ambient_source_is_not_replaced_by_repeated_impersonation_failures`. See
    /// `dead_federated_source_is_replaced_after_permanent_mint_failure`'s doc for why the outer
    /// credential must be a `LeasedMockSource` (a real lease), not a plain `MockSource`.
    #[restate_core::test]
    async fn healthy_federated_source_is_not_replaced_by_repeated_mint_failures() {
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/stable";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let registry = credential_registry_for_test();
        let access_token_source = federation::federated_access_token_source(
            &registry.federated_access_token_sources,
            provider,
        );
        access_token_source
            .credentials
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
            ))
            .await;

        let build_count = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider, {
            let build_count = build_count.clone();
            move || {
                build_count.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let audience = "https://federated-stable.example.com";
        let spec = IdTokenSpec::federated(audience, provider, service_account);
        add_build_override(spec.clone(), {
            let access_token_source = access_token_source.clone();
            move |spec| {
                Ok(Arc::new(LeasedMockSource {
                    spec: spec.clone(),
                    _access_token_source: access_token_source.clone(),
                }) as Arc<dyn IdTokenSource>)
            }
        });
        drop(access_token_source);

        let client = GcpTokenClient::new(TaskCenter::current());
        for _ in 0..5 {
            let outcome = client
                .mint(Some(provider), Some(service_account), audience)
                .await;
            assert!(
                matches!(outcome, Err(GcpAuthError::Mint { .. })),
                "{outcome:?}"
            );
        }

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            0,
            "a healthy federated source must never be replaced by an impersonation-only failure"
        );
    }

    /// Two distinct WIF providers get two independent shared source credentials: a permanent
    /// failure on a key for one provider must never touch the other's source, and each provider's
    /// source builds independently. See `dead_federated_source_is_replaced_after_permanent_mint_failure`'s
    /// doc for why the outer credentials are `LeasedMockSource` (a real lease), not plain
    /// `MockSource`.
    #[restate_core::test]
    async fn two_wif_providers_get_independent_sources() {
        let provider_a =
            "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/a";
        let provider_b =
            "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/b";
        let registry = credential_registry_for_test();
        let access_token_source_a = federation::federated_access_token_source(
            &registry.federated_access_token_sources,
            provider_a,
        );
        access_token_source_a
            .credentials
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Dead),
            ))
            .await;
        let access_token_source_b = federation::federated_access_token_source(
            &registry.federated_access_token_sources,
            provider_b,
        );
        access_token_source_b
            .credentials
            .seed_for_test(google_cloud_auth::credentials::Credentials::from(
                FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
            ))
            .await;

        let build_count_a = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider_a, {
            let build_count_a = build_count_a.clone();
            move || {
                build_count_a.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });
        let build_count_b = Arc::new(AtomicUsize::new(0));
        federation::install_federated_access_token_source_override_for_test(provider_b, {
            let build_count_b = build_count_b.clone();
            move || {
                build_count_b.fetch_add(1, Ordering::SeqCst);
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            }
        });

        let client = GcpTokenClient::new(TaskCenter::current());
        let service_account = "sa@example.iam.gserviceaccount.com";
        let audience_a = "https://federated-independent-a.example.com";
        let audience_b = "https://federated-independent-b.example.com";
        let spec_a = IdTokenSpec::federated(audience_a, provider_a, service_account);
        let spec_b = IdTokenSpec::federated(audience_b, provider_b, service_account);
        add_build_override(spec_a.clone(), {
            let access_token_source_a = access_token_source_a.clone();
            move |spec| {
                Ok(Arc::new(LeasedMockSource {
                    spec: spec.clone(),
                    _access_token_source: access_token_source_a.clone(),
                }) as Arc<dyn IdTokenSource>)
            }
        });
        add_build_override(spec_b.clone(), {
            let access_token_source_b = access_token_source_b.clone();
            move |spec| {
                Ok(Arc::new(LeasedMockSource {
                    spec: spec.clone(),
                    _access_token_source: access_token_source_b.clone(),
                }) as Arc<dyn IdTokenSource>)
            }
        });
        drop(access_token_source_a);
        drop(access_token_source_b);

        // Only provider_a's source is dead; failing a mint against provider_b must never touch it.
        let outcome_b = client
            .mint(Some(provider_b), Some(service_account), audience_b)
            .await;
        assert!(matches!(outcome_b, Err(GcpAuthError::Mint { .. })));
        assert_eq!(
            build_count_b.load(Ordering::SeqCst),
            0,
            "provider_b's healthy source must not be replaced"
        );
        assert_eq!(
            build_count_a.load(Ordering::SeqCst),
            0,
            "a mint against provider_b must never rebuild provider_a's source"
        );

        // provider_a's dead source is replaced on its own key's permanent failure.
        let outcome_a = client
            .mint(Some(provider_a), Some(service_account), audience_a)
            .await;
        assert!(matches!(outcome_a, Err(GcpAuthError::Mint { .. })));
        assert_eq!(
            build_count_a.load(Ordering::SeqCst),
            1,
            "provider_a's dead source must be replaced exactly once"
        );
        assert_eq!(
            build_count_b.load(Ordering::SeqCst),
            0,
            "recovering provider_a's source must never touch provider_b's"
        );
    }

    /// `wif_provider` is a real dimension of the cache key: a federated and an ambient/impersonated
    /// credential for the same (impersonate, audience) pair must never collide in the registry.
    /// Drives this through the real `mint()` -> `get_or_build` path via `build_overrides` (each key
    /// gets its own override, keyed exactly as production would key it), rather than manipulating
    /// the cache directly, so a regression that merged the two keys would actually fail this test.
    #[restate_core::test]
    async fn wif_provider_is_a_distinct_cache_key_dimension() {
        let client = GcpTokenClient::new(TaskCenter::current());
        let audience = "https://wif-cache-key.example.com";
        let impersonate = "sa@proj.iam.gserviceaccount.com";
        let provider =
            "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/r";

        let ambient_key = IdTokenSpec {
            wif_provider: None,
            impersonate: Some(impersonate.to_owned()),
            audience: audience.to_owned(),
        };
        let wif_key = IdTokenSpec {
            wif_provider: Some(provider.to_owned()),
            impersonate: Some(impersonate.to_owned()),
            audience: audience.to_owned(),
        };
        assert_ne!(ambient_key, wif_key);

        let ambient_builds = Arc::new(AtomicUsize::new(0));
        let wif_builds = Arc::new(AtomicUsize::new(0));

        add_build_override(ambient_key, {
            let ambient_builds = ambient_builds.clone();
            move |_| {
                ambient_builds.fetch_add(1, Ordering::SeqCst);
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            }
        });
        add_build_override(wif_key, {
            let wif_builds = wif_builds.clone();
            move |_| {
                wif_builds.fetch_add(1, Ordering::SeqCst);
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            }
        });

        // Minting through each key twice must build each exactly once (moka caches the result)
        // and never satisfy one key's construction from the other's.
        for _ in 0..2 {
            client
                .mint(None, Some(impersonate), audience)
                .await
                .expect("ambient key mints");
            client
                .mint(Some(provider), Some(impersonate), audience)
                .await
                .expect("federated key mints");
        }

        assert_eq!(
            ambient_builds.load(Ordering::SeqCst),
            1,
            "ambient key must build exactly once, independently of the federated key"
        );
        assert_eq!(
            wif_builds.load(Ordering::SeqCst),
            1,
            "federated key must build exactly once, independently of the ambient key"
        );
    }

    /// A deployment requesting workload identity federation on a server with no `[gcp-federation]`
    /// configuration must fail closed with a construction error, never fall back to an
    /// unauthenticated request. This exercises the real `mint` -> registry -> construction path
    /// (not a seeded test source), with `FEDERATION_CONFIG` left unset -- nothing in this test
    /// binary ever installs a `[gcp-federation]` config (see `federation::federation_tests` for
    /// why that must stay true).
    ///
    /// Construction failures are never cached by moka's `try_get_with` (only successful builds
    /// are), so there is no stale entry to evict here -- unlike a *cached* credential's permanent
    /// mint failure, which does go through `evict_if_unchanged`. Retrying finds nothing cached and
    /// attempts construction fresh, which is the PR #1 discipline this case trivially satisfies.
    #[restate_core::test]
    async fn wif_requested_without_server_config_is_a_permanent_build_error() {
        let client = GcpTokenClient::new(TaskCenter::current());
        let provider =
            "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/r";
        let audience = "https://wif-no-config.example.com";
        let impersonate = "sa@proj.iam.gserviceaccount.com";

        let err = client
            .mint(Some(provider), Some(impersonate), audience)
            .await
            .expect_err("must fail without a [gcp-federation] configuration");
        assert!(
            matches!(err, GcpAuthError::Adc { .. }),
            "expected a construction error, got {err:?}"
        );

        let key = IdTokenSpec {
            wif_provider: Some(provider.to_owned()),
            impersonate: Some(impersonate.to_owned()),
            audience: audience.to_owned(),
        };
        assert!(
            credential_registry_for_test().cache.get(&key).await.is_none(),
            "a construction failure must never populate the cache"
        );

        // Retrying attempts construction fresh and fails the same way -- not wedged on a stale
        // cached error.
        let err2 = client
            .mint(Some(provider), Some(impersonate), audience)
            .await
            .expect_err("still fails without configuration");
        assert!(matches!(err2, GcpAuthError::Adc { .. }));
    }

    /// The registry must not survive its task center. Embedded Restate creates and destroys task
    /// centers within a process (`Restate::create`/`Restate::stop`); a registry's cache and
    /// ambient source are only meaningful for the task center whose default runtime hosts their
    /// housekeeping task and construction tasks -- once that task center shuts down, a later
    /// `mint()` under a *new* task center must get a fresh registry, not the orphaned one.
    #[tokio::test(flavor = "multi_thread")]
    async fn registry_rebuilds_after_the_task_center_that_built_it_is_replaced() {
        use restate_core::{TaskCenterBuilder, TaskCenterFutureExt as _};

        let audience = "https://tc-lifecycle.example.com";
        let spec = IdTokenSpec::ambient(audience);
        let build_count = Arc::new(AtomicUsize::new(0));

        // Re-installed under each task center in turn: each gets its own fresh `CredentialRegistry`
        // (and so its own, empty `TestHooks`), so the override must be added again for it to see it.
        let install_override = || {
            add_build_override(spec.clone(), {
                let build_count = build_count.clone();
                move |_| {
                    build_count.fetch_add(1, Ordering::SeqCst);
                    Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
                }
            });
        };

        let tc_a = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();
        async {
            install_override();
            let registry = credential_registry(&tc_a).expect("tc_a is not shutting down");
            let result = registry.get_or_build(&spec).await;
            if let Err(error) = &result {
                panic!("{error}");
            }
        }
        .in_tc(&tc_a)
        .await;
        assert_eq!(build_count.load(Ordering::SeqCst), 1);
        tc_a.shutdown_node("test done with TC-A", 0).await;

        let tc_b = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();
        async {
            install_override();
            let registry = credential_registry(&tc_b).expect("tc_b is not shutting down");
            let result = registry.get_or_build(&spec).await;
            if let Err(error) = &result {
                panic!("{error}");
            }
        }
        .in_tc(&tc_b)
        .await;
        assert_eq!(
            build_count.load(Ordering::SeqCst),
            2,
            "a new task center must get a fresh registry -- the old one's cache must not persist"
        );
        tc_b.shutdown_node("test done with TC-B", 0).await;
    }

    /// The federated analogue of `registry_rebuilds_after_the_task_center_that_built_it_is_replaced`:
    /// a federated key's per-provider access-token sources live on `CredentialRegistry` (not a
    /// module-level static) for exactly the same reason the outer `cache`/`ambient_source` do --
    /// a source's refresh task is spawned on whichever task center's default runtime built it, so
    /// it must rebuild, not reuse, when a new task center replaces the old one.
    #[tokio::test(flavor = "multi_thread")]
    async fn federated_source_rebuilds_after_the_task_center_that_built_it_is_replaced() {
        use restate_core::{TaskCenterBuilder, TaskCenterFutureExt as _};

        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/tc-lifecycle";
        let audience = "https://tc-lifecycle-federated.example.com";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let spec = IdTokenSpec::federated(audience, provider, service_account);
        let build_count = Arc::new(AtomicUsize::new(0));

        // Reinstalled under each task center in turn -- unlike `build_overrides`, this override
        // lives in a plain federation-module static (see `install_federated_access_token_source_override_for_test`),
        // so it would already persist across task centers on its own; re-installing here just
        // keeps the two lifecycle tests symmetric and makes no difference to the outcome.
        let install_override = || {
            federation::install_federated_access_token_source_override_for_test(provider, {
                let build_count = build_count.clone();
                move || {
                    build_count.fetch_add(1, Ordering::SeqCst);
                    Ok(google_cloud_auth::credentials::Credentials::from(
                        FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                    ))
                }
            });
        };

        let tc_a = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();
        async {
            install_override();
            // The outer credential build is real (not overridden), so it may fail; only the
            // source-sharing build count below is under test.
            let _ = credential_registry(&tc_a).expect("tc_a is not shutting down").get_or_build(&spec).await;
        }
        .in_tc(&tc_a)
        .await;
        assert_eq!(build_count.load(Ordering::SeqCst), 1);
        tc_a.shutdown_node("test done with TC-A", 0).await;

        let tc_b = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();
        async {
            install_override();
            let _ = credential_registry(&tc_b).expect("tc_b is not shutting down").get_or_build(&spec).await;
        }
        .in_tc(&tc_b)
        .await;
        assert_eq!(
            build_count.load(Ordering::SeqCst),
            2,
            "a new task center must rebuild the federated access-token source too -- the old \
             registry's federated_access_token_sources map must not persist"
        );
        tc_b.shutdown_node("test done with TC-B", 0).await;
    }

    /// Complements the two rebuild tests above: those prove a *new* task center gets a fresh
    /// registry; this one proves the *old* registry actually dies at shutdown, not merely whenever
    /// a later mint happens to replace it. `ClearRegistrySlotOnDrop` lives inside the housekeeping
    /// future specifically so `TaskKind::Credentials`'s `OnCancel = "abort"` shutdown path drops
    /// it.
    ///
    /// Builds a real federated outer credential via `get_or_build` (the same path `mint()` uses)
    /// and takes its access-token source's `Weak` from the registry's own map, rather than
    /// seeding a source directly and taking a `Weak` from a locally-owned `Arc` -- that would
    /// prove nothing about TaskCenter shutdown, only about the test's own variable going out of
    /// scope (see [`federation::FederatedAccessTokenSources`]'s doc for why only a cached outer
    /// credential's lease legitimately keeps a source alive).
    #[tokio::test(flavor = "multi_thread")]
    async fn registry_is_dropped_at_shutdown_not_at_the_next_mint() {
        use restate_core::{TaskCenterBuilder, TaskCenterFutureExt as _};

        let audience = "https://tc-shutdown-drop.example.com";
        let spec = IdTokenSpec::ambient(audience);
        let provider = "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/tc-shutdown-drop";
        let service_account = "sa@example.iam.gserviceaccount.com";
        let federated_spec = IdTokenSpec::federated(
            "https://tc-shutdown-drop-federated.example.com",
            provider,
            service_account,
        );

        let tc = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();

        let (weak, federated_source_weak) = async {
            add_build_override(spec.clone(), |_| {
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            });
            federation::install_federated_access_token_source_override_for_test(provider, || {
                Ok(google_cloud_auth::credentials::Credentials::from(
                    FakeCredentialsProvider::always(|| ProbeOutcome::Healthy),
                ))
            });

            let registry = credential_registry(&tc).expect("tc is not shutting down");
            let result = registry.get_or_build(&spec).await;
            if let Err(error) = &result {
                panic!("{error}");
            }
            // Caches a real `FederatedIdTokenCredentials` -- the same path `mint()` uses for a
            // cold federated key -- so the weak entry taken below is leased by something genuine,
            // not by a local variable this test itself would be the sole owner of.
            let federated_result = registry.get_or_build(&federated_spec).await;
            if let Err(error) = &federated_result {
                panic!("{error}");
            }
            let federated_source_weak = registry
                .federated_access_token_sources
                .lock()
                .get(provider)
                .cloned()
                .expect(
                    "the federated build above must have created a weak entry for this provider",
                );
            (Arc::downgrade(&registry), federated_source_weak)
        }
        .in_tc(&tc)
        .await;

        // Before shutdown, the access-token source must be alive: it's leased by the outer
        // federated credential `get_or_build` just cached.
        assert!(
            federated_source_weak.upgrade().is_some(),
            "the federated access-token source must be alive before TaskCenter shutdown"
        );

        tc.shutdown_node("test done", 0).await;

        // shutdown_node's cancel_tasks aborts an OnCancel="abort" task without awaiting its
        // teardown (tokio's `JoinHandle::abort` only guarantees the task is dropped at its next
        // poll, not synchronously with the call), so completion of shutdown_node does not
        // strictly happen-after the guard's Drop. Poll for it instead of asserting immediately.
        // Checking `strong_count() == 0` rather than `upgrade().is_none()` matters here: holding
        // an `upgrade()`d Arc across the sleep would itself keep the registry alive, making the
        // loop self-perpetuating.
        let mut dropped = weak.strong_count() == 0;
        for _ in 0..50 {
            if dropped {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            dropped = weak.strong_count() == 0;
        }
        assert!(
            dropped,
            "the registry must be dropped at TaskCenter shutdown, not at the next mint"
        );
        assert!(
            federated_source_weak.upgrade().is_none(),
            "the federated access-token source must die with the registry, outer cache, and \
             cached outer credential that leased it"
        );
    }

    /// The property construction via TaskCenter exists to provide: a real credential's `build()`
    /// spawns a background refresh task via a bare `tokio::spawn`, so that refresh task must land
    /// on a runtime with process lifetime, not whatever runtime happened to call `mint()` --
    /// otherwise dropping a partition's own runtime would silently kill every credential's
    /// refresh task built while handling that partition.
    ///
    /// Builds its own `TaskCenter` with its own dedicated default runtime (standing in for the
    /// server's real one), then drives construction from a separate, disposable "caller" runtime
    /// that is dropped once construction returns. The build override -- now consulted *inside*
    /// the spawned `TaskKind::Credentials` task, see `build_on_tc_task` -- stands in for a real
    /// credential's `build()`: it spawns a probe child task and returns. The probe surviving the
    /// caller runtime's drop proves it (and by extension any real `build()`) ran on TaskCenter's
    /// default runtime, not the caller's.
    #[test]
    fn credential_construction_runs_on_task_centers_default_runtime_not_the_callers() {
        use restate_core::TaskCenterFutureExt as _;

        let default_runtime = tokio::runtime::Runtime::new().expect("default runtime builds");
        let task_center = restate_core::TaskCenterBuilder::default()
            .default_runtime_handle(default_runtime.handle().clone())
            .build()
            .expect("task center builds")
            .into_handle();

        let audience = "https://runtime-affinity.example.com";
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let probe_completed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        task_center.run_sync(|| {
            let running = running.clone();
            let probe_completed = probe_completed.clone();
            add_build_override(IdTokenSpec::ambient(audience), move |_| {
                let running = running.clone();
                let probe_completed = probe_completed.clone();
                tokio::spawn(async move {
                    while running.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    probe_completed.store(true, Ordering::SeqCst);
                });
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            });
        });

        {
            let caller_runtime = tokio::runtime::Runtime::new().expect("caller runtime builds");
            let spec = IdTokenSpec::ambient(audience);
            let registry =
                credential_registry(&task_center).expect("task_center is not shutting down");
            let result = caller_runtime
                .block_on(async { registry.get_or_build(&spec).await }.in_tc(&task_center));
            if let Err(error) = &result {
                panic!("{error}");
            }
            // Dropping the caller runtime here. If the probe had landed on it instead of
            // TaskCenter's default runtime, dropping it would abort the probe before it can ever
            // observe `running` flip to false.
        }

        std::thread::sleep(Duration::from_millis(20));
        running.store(false, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            probe_completed.load(Ordering::SeqCst),
            "a task spawned during construction must survive the caller runtime's drop"
        );
    }

    /// A burst of concurrent distinct-key blocking builds is bounded by `BLOCKING_BUILD_PERMITS`,
    /// not by tokio's much larger default blocking-thread-pool size -- proving the bound is
    /// actually enforced, not merely declared, and that exceeding it does not deadlock.
    ///
    /// Drives concurrency directly through `run_blocking`, the production choke point both outer
    /// and ambient-source construction share, rather than through `mint()`: the build override
    /// consulted in `build_on_tc_task` (see the runtime-affinity test above) sits above
    /// `run_blocking` and would bypass it entirely, so routing this test through `mint()` would
    /// not exercise the semaphore at all. `run_blocking` is private but reachable directly from
    /// this submodule, which is the least intrusive way to exercise it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn blocking_builds_are_bounded_and_never_deadlock() {
        let concurrent = Arc::new(AtomicUsize::new(0));
        let high_water_mark = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..4 * MAX_CONCURRENT_BLOCKING_BUILDS)
            .map(|_| {
                let concurrent = concurrent.clone();
                let high_water_mark = high_water_mark.clone();
                tokio::spawn(run_blocking("test".to_owned(), move || {
                    let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                    high_water_mark.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(20));
                    concurrent.fetch_sub(1, Ordering::SeqCst);
                    Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
                }))
            })
            .collect();

        for task in tasks {
            let result = task.await.expect("task doesn't panic");
            if let Err(error) = &result {
                panic!("{error}");
            }
        }

        let mark = high_water_mark.load(Ordering::SeqCst);
        assert!(
            mark <= MAX_CONCURRENT_BLOCKING_BUILDS,
            "at most {MAX_CONCURRENT_BLOCKING_BUILDS} blocking builds may run concurrently, saw {mark}"
        );
    }

    /// Invoker tasks run on a plain `tokio::JoinSet` without a TaskCenter task-local. Build the
    /// client under TaskCenter A, mint from a bare task on another runtime, and verify credential
    /// construction still runs on A's default runtime.
    ///
    /// This uses hand-built runtimes because Tokio forbids dropping a runtime from another
    /// runtime's async context.
    #[test]
    fn mint_succeeds_from_a_task_with_no_task_center_task_local() {
        use restate_core::TaskCenterBuilder;

        let default_runtime = tokio::runtime::Runtime::new().expect("default runtime builds");
        let task_center = TaskCenterBuilder::default()
            .default_runtime_handle(default_runtime.handle().clone())
            .build()
            .expect("task center builds")
            .into_handle();

        let audience = "https://no-task-local.example.com";
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let probe_completed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Cold cache: this is the first thing this fresh TaskCenter's registry ever builds.
        task_center.run_sync(|| {
            let running = running.clone();
            let probe_completed = probe_completed.clone();
            add_build_override(IdTokenSpec::ambient(audience), move |_| {
                let running = running.clone();
                let probe_completed = probe_completed.clone();
                tokio::spawn(async move {
                    while running.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    probe_completed.store(true, Ordering::SeqCst);
                });
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            });
        });

        let client = GcpTokenClient::new(task_center);

        {
            // Stands in for an invoker invocation task's plain `tokio::JoinSet`: a runtime with no
            // relation to `task_center`, on which `tokio::spawn` produces a task with no
            // `TaskCenter` task-local at all.
            let invocation_runtime = tokio::runtime::Runtime::new().expect("runtime builds");
            let mint_result = invocation_runtime.block_on(async {
                tokio::spawn(async move {
                    assert!(
                        TaskCenter::try_current().is_none(),
                        "this task must carry no TaskCenter task-local, to reproduce the invoker's plain JoinSet"
                    );
                    client.mint(None, None, audience).await
                })
                .await
                .expect("mint task must not panic")
            });
            assert!(mint_result.is_ok(), "{mint_result:?}");
            // Dropping the invocation runtime here, before checking the probe below: if
            // construction had landed on it instead of TaskCenter A's own default runtime, this
            // drop would abort the probe before it can ever observe `running` flip to false.
        }

        // The credential build itself still ran as a `TaskKind::Credentials` task on TaskCenter
        // A's own default runtime -- via the `Handle` stored on the client, never a task-local.
        std::thread::sleep(Duration::from_millis(20));
        running.store(false, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            probe_completed.load(Ordering::SeqCst),
            "credential construction must run on TaskCenter A's own default runtime"
        );
    }

    /// A client retained across an embedded-server restart must fail cleanly without reinstalling
    /// its stopped TaskCenter's registry over the new generation.
    #[tokio::test(flavor = "multi_thread")]
    async fn stale_client_bound_to_a_shutdown_task_center_cannot_displace_the_new_generation() {
        use restate_core::{TaskCenterBuilder, TaskCenterFutureExt as _};

        let audience = "https://tc-generation.example.com";

        let tc_a = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();
        let stale_client = GcpTokenClient::new(tc_a.clone());
        let warm = async {
            add_build_override(IdTokenSpec::ambient(audience), |_| {
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            });
            stale_client.mint(None, None, audience).await
        }
        .in_tc(&tc_a)
        .await;
        assert!(warm.is_ok(), "{warm:?}");

        tc_a.shutdown_node("test done with TC-A", 0).await;

        let tc_b = TaskCenterBuilder::default_for_tests()
            .build()
            .expect("task center builds")
            .into_handle();
        let fresh_client = GcpTokenClient::new(tc_b.clone());
        let fresh = async {
            add_build_override(IdTokenSpec::ambient(audience), |_| {
                Ok(MockSource::new(|_| MockOutcome::Token(token())) as Arc<dyn IdTokenSource>)
            });
            fresh_client.mint(None, None, audience).await
        }
        .in_tc(&tc_b)
        .await;
        assert!(fresh.is_ok(), "{fresh:?}");

        let b_registry = credential_registry(&tc_b).expect("tc_b is not shutting down");

        // tc_a is shut down; the stale client's mint must surface a clean build error, not panic,
        // and must not install a dead TC-A registry over the slot TC B occupies.
        let stale_after_shutdown = stale_client.mint(None, None, audience).await;
        assert!(
            matches!(stale_after_shutdown, Err(GcpAuthError::Build { .. })),
            "{stale_after_shutdown:?}"
        );

        let b_registry_after = credential_registry(&tc_b).expect("tc_b is not shutting down");
        assert!(
            Arc::ptr_eq(&b_registry, &b_registry_after),
            "a stale, shut-down client's mint must not displace TC B's registry slot"
        );

        tc_b.shutdown_node("test done with TC-B", 0).await;
    }
}
