# Dev v3.4 Status Field Forensics

- Slots: 27
- Raw statuses: `{'answerable': 3, 'supported': 23, 'unsupported': 1}`
- Status-excluded structural fields valid: 27/27
- New unique content shapes already valid: 0/27
- Supported with empty claim text: 0
- Supported with non-empty omission reason: 2

All 27 slots retain the expected field names, types, and required-claim IDs apart from the semantic status enum. However, none already satisfies the new unique content-shape contract: most answered-looking slots use an exact empty omission_reason rather than null, while q015 contains claim/reason conflicts. Status removal alone therefore does not make the nine answerable historical payloads valid.
