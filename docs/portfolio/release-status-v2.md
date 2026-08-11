# Portfolio Release Status v2

## Current status

| Field | Value |
| --- | --- |
| Latest released tag | `v1.1.0-portfolio` |
| Package/runtime version | `1.1.0+portfolio` |
| Candidate under audit | `v1.2.0-portfolio` |
| Version bump in this branch | `false` |
| Tag creation in this branch | `false` |
| GitHub Release creation | `false` |

## v1.2.0 candidate scope

The v1.2.0 candidate is a release-readiness and public-documentation truth audit
for changes after `v1.1.0-portfolio`:

- Research UI mode separation for Workflow and Agent.
- Agent final-report synthesis from verified Evidence State.
- UI result alignment with runtime outputs.
- Linux CI portability repairs.
- GitHub Actions runtime maintenance.

## Release interpretation

`READY_WITH_DOCUMENTED_LIMITATION` is the expected readiness class if local and
remote gates pass. The documented limitation is that Stage 4 benchmark results
remain historical v1.1.0 artifacts and do not include the later Agent
final-report synthesis layer.

## Not authorized here

- No v1.2.0 tag.
- No GitHub Release.
- No version bump.
- No rerun of Stage 4 benchmark.
- No live provider calls.
