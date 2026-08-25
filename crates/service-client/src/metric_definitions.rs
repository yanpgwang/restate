// Copyright (c) 2023 - 2026 Restate Software, Inc., Restate GmbH.
// All rights reserved.
//
// Use of this software is governed by the Business Source License
// included in the LICENSE file.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0.

use metrics::{Unit, describe_counter, describe_gauge, describe_histogram};

pub(crate) const GCP_CREDENTIAL_BUILDS: &str = "restate.service_client.gcp.credential_builds.total";
pub(crate) const GCP_CREDENTIAL_BUILD_DURATION: &str =
    "restate.service_client.gcp.credential_build_duration.seconds";
pub(crate) const GCP_TOKEN_MINTS: &str = "restate.service_client.gcp.token_mints.total";
pub(crate) const GCP_CREDENTIALS_ACTIVE: &str = "restate.service_client.gcp.credentials.active";

/// Incremented at `AwsSubjectTokenProvider::subject_token()`'s single exit. `google-cloud-auth`'s
/// external-account refresh loop calls this on every federated credential refresh, so it is the
/// only crate-external heartbeat of federated refresh activity available without upstream
/// support: its success rate approximates federated refresh health, and its errors surface
/// failures in the AWS hop (broker credential fetch and SigV4 signing) specifically, ahead of the
/// STS exchange and impersonation steps that follow.
pub(crate) const GCP_FEDERATION_SUBJECT_TOKENS: &str =
    "restate.service_client.gcp.federation.subject_tokens.total";
/// The per-provider twin of [`GCP_CREDENTIALS_ACTIVE`]: the population of shared federated
/// access-token sources, one per WIF provider resource in `federated_access_token_sources`. Set
/// only on the registry's housekeeping tick, after that tick's own reap pass -- unlike
/// `GCP_CREDENTIALS_ACTIVE`, there is no separate after-each-build update site any more: a
/// provider's access-token source is weak-indexed and leased by every outer credential that
/// references it, not time-evicted, so its count can only meaningfully change when housekeeping
/// prunes dead leases, not on every build. Approximate for the same reason
/// `GCP_CREDENTIALS_ACTIVE` is (and then some): an access-token source can go fully unreferenced
/// and still count as active here until the next housekeeping tick reaps it, so this can lag an
/// actual removal by up to one housekeeping interval.
pub(crate) const GCP_FEDERATION_SOURCES_ACTIVE: &str =
    "restate.service_client.gcp.federation.sources.active";

pub(crate) const RESULT_SUCCESS: &str = "success";
pub(crate) const RESULT_ERROR: &str = "error";

pub(crate) const MINT_OUTCOME_SUCCESS: &str = "success";
pub(crate) const MINT_OUTCOME_TIMEOUT: &str = "timeout";
pub(crate) const MINT_OUTCOME_TRANSIENT_ERROR: &str = "transient_error";
pub(crate) const MINT_OUTCOME_PERMANENT_ERROR: &str = "permanent_error";
pub(crate) const MINT_OUTCOME_BUILD_ERROR: &str = "build_error";

/// `token_mints.total`'s `mode` label: separates federated mint failures (the customer-facing
/// misconfiguration surface for Cloud -- a wrong WIF provider or missing impersonation binding)
/// from ordinary ADC-path failures, without log-diving. Deliberately not added to the build
/// counters: a build failure is already distinguishable from its log record, and duplicating the
/// dimension there buys nothing.
pub(crate) const MINT_MODE_ADC: &str = "adc";
pub(crate) const MINT_MODE_FEDERATED: &str = "federated";

/// Label values only -- never audience, service account, endpoint, provider, or error text.
/// Those are unbounded, and the credential registry these metrics describe is a small,
/// process-wide cache, not a per-target one.
pub(crate) fn describe_metrics() {
    describe_counter!(
        GCP_CREDENTIAL_BUILDS,
        Unit::Count,
        "Number of GCP credential build attempts, by result"
    );

    describe_histogram!(
        GCP_CREDENTIAL_BUILD_DURATION,
        Unit::Seconds,
        "Time to build a GCP credential, including ambient source resolution"
    );

    describe_counter!(
        GCP_TOKEN_MINTS,
        Unit::Count,
        "Number of GCP ID-token mint attempts, by outcome (success, timeout, transient_error, \
         permanent_error -- the credential's own id_token() call failed, or build_error -- the \
         underlying credential itself could not be constructed, counted once per failed caller, \
         not once per build, since a single failed build fails every caller waiting on it) and \
         mode (adc or federated)"
    );

    describe_gauge!(
        GCP_CREDENTIALS_ACTIVE,
        Unit::Count,
        "Number of GCP credentials currently cached. Approximate: moka evicts lazily, so this can \
         lag an actual eviction until the next housekeeping tick"
    );

    describe_counter!(
        GCP_FEDERATION_SUBJECT_TOKENS,
        Unit::Count,
        "Number of AWS subject-token fetches for GCP workload identity federation, by result"
    );

    describe_gauge!(
        GCP_FEDERATION_SOURCES_ACTIVE,
        Unit::Count,
        "Number of GCP workload identity federation access-token sources currently live, one per \
         provider. Approximate: leased by outer credentials rather than time-evicted, and updated \
         only on the housekeeping tick, so this can lag an actual removal by up to one \
         housekeeping interval"
    );
}
