# Dev v3.2 citation improvement plan v1

Offline design only. No live code or model call was executed.

- **primary citation first**: Prefer the strongest claim-local evidence before supplements. (affected claims: 8)
- **per-claim citation cap**: Limit dilution from weak extra citations. (affected claims: 8)
- **compound claim decomposition**: Split claims whose minimum evidence set spans subclaims. (affected claims: 3)
- **numeric evidence completeness validator**: Require cited text to contain required numeric facts. (affected claims: 2)
- **comparison-side completeness validator**: Require evidence for every comparison side. (affected claims: 4)
- **core-set-aware evidence allocation**: Allocate all members of a minimum complete evidence set. (affected claims: 10)
- **original evidence first**: Use adjacent completion only as supporting context. (affected claims: 4)
- **shrink claim or return unsupported**: Avoid unsupported breadth when evidence is incomplete. (affected claims: 4)

All candidates are generic, have no online Gold/human-label dependency, and require explicitly authorized live validation.
