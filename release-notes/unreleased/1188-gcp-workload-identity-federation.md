# AWS-to-GCP workload identity federation

## New Feature

### What Changed

AWS-hosted Restate servers can now authenticate HTTP deployments to private Cloud Run services
without storing Google credentials. The authentication chain is:

```text
ambient AWS identity
  -> assume the operator-configured AWS federation role
  -> sign an AWS GetCallerIdentity subject token
  -> exchange it through the deployment's Google workload identity provider
  -> impersonate the deployment's Google service account to mint an ID token
```

The assumed AWS role session is shared across the process. Google access-token sources are shared
per workload identity provider and remain live only while cached ID-token credentials reference
them.

### Configuration

Server operators enable the feature under the invoker's service-client options:

```toml
[worker.invoker.service-client.gcp-federation]
aws-role-arn = "arn:aws:iam::<account>:role/<federation-role>"
aws-role-session-name = "<a value allowed by the role's trust policy>"
```

A deployment then opts in at registration time:

```
restate dp register https://svc-abc-uc.a.run.app \
  --gcp-impersonate-service-account caller@project.iam.gserviceaccount.com \
  --gcp-workload-identity-provider "//iam.googleapis.com/projects/N/locations/global/workloadIdentityPools/P/providers/R"
```

`--gcp-workload-identity-provider` requires `--gcp-impersonate-service-account`. A deployment that
requests federation on a server with no `gcp-federation` configured fails registration and every
subsequent mint attempt with an actionable error -- never an unauthenticated fallback request.

The `gcp-federation` block is not live-reloadable. Changing `aws-role-arn` or
`aws-role-session-name` requires a server restart.
