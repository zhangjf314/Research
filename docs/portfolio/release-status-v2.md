# Portfolio Release Status v2

## Current status

| Field | Value |
| --- | --- |
| Latest released tag | `v1.2.0-portfolio` |
| Package/runtime version | `1.2.0+portfolio` |
| Release status | `RELEASED_WITH_DOCUMENTED_LIMITATIONS` |
| Version bump in release commit | `true` |
| Tag creation | `authorized after green main CI` |
| GitHub Release creation | `authorized after tag verification` |

## v1.2.0 release scope

The v1.2.0 release covers audited changes after `v1.1.0-portfolio`:

- Research UI mode separation for Workflow and Agent.
- Agent final-report synthesis from verified Evidence State.
- UI result alignment with runtime outputs.
- Linux CI portability repairs.
- GitHub Actions runtime maintenance.

## Release interpretation

`RELEASED_WITH_DOCUMENTED_LIMITATIONS` is the release class. The documented
limitation is that Stage 4 benchmark results remain historical v1.1.0 artifacts
and do not include the later Agent final-report synthesis layer.

## Not included

- No rerun of Stage 4 benchmark.
- No live provider calls.
