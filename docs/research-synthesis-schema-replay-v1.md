# Research synthesis schema replay v1

```json
{
  "schema_version": "research-synthesis-schema-replay-v1",
  "input": ".runtime/research-synthesis-provider/c88e94e6-a8b1-41d5-8c6f-b75cc90778db",
  "attempts": [
    {
      "attempt_number": 1,
      "content_length": 4689,
      "json_parse_status": "passed",
      "normalization_actions": [],
      "top_level_keys": [
        "consensus",
        "disagreements",
        "executive_summary",
        "research_gaps",
        "sections",
        "title"
      ],
      "schema_error_count": 1,
      "schema_error_locations": [
        "<root>"
      ],
      "schema_error_types": [
        "ValueError"
      ],
      "failure_types": [
        "CITATION_NOT_ALLOWED_FOR_SECTION"
      ],
      "offending_citation_ids": [
        "E02",
        "E07",
        "E15",
        "E16"
      ],
      "citation_allowlist_details": [
        {
          "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
          "location": "sections.0.claims.2.citation_ids",
          "section_id": "background",
          "citation_ids": [
            "E16"
          ],
          "allowed_for_section": [
            "E01",
            "E02",
            "E03",
            "E04",
            "E05",
            "E06",
            "E07",
            "E08"
          ]
        },
        {
          "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
          "location": "sections.0.claims.3.citation_ids",
          "section_id": "background",
          "citation_ids": [
            "E15"
          ],
          "allowed_for_section": [
            "E01",
            "E02",
            "E03",
            "E04",
            "E05",
            "E06",
            "E07",
            "E08"
          ]
        },
        {
          "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
          "location": "sections.0.claims.4.citation_ids",
          "section_id": "background",
          "citation_ids": [
            "E16"
          ],
          "allowed_for_section": [
            "E01",
            "E02",
            "E03",
            "E04",
            "E05",
            "E06",
            "E07",
            "E08"
          ]
        },
        {
          "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
          "location": "sections.1.claims.3.citation_ids",
          "section_id": "methods",
          "citation_ids": [
            "E07"
          ],
          "allowed_for_section": [
            "E01",
            "E04",
            "E05",
            "E06",
            "E09",
            "E10",
            "E11",
            "E12"
          ]
        },
        {
          "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
          "location": "sections.2.claims.2.citation_ids",
          "section_id": "results",
          "citation_ids": [
            "E02"
          ],
          "allowed_for_section": [
            "E01",
            "E03",
            "E04",
            "E06",
            "E07",
            "E08",
            "E12",
            "E13"
          ]
        }
      ],
      "research_synthesis_schema": "failed",
      "citation_allowlist": "failed"
    },
    {
      "attempt_number": 2,
      "content_length": 4689,
      "json_parse_status": "passed",
      "normalization_actions": [],
      "top_level_keys": [
        "consensus",
        "disagreements",
        "executive_summary",
        "research_gaps",
        "sections",
        "title"
      ],
      "schema_error_count": 1,
      "schema_error_locations": [
        "<root>"
      ],
      "schema_error_types": [
        "ValueError"
      ],
      "failure_types": [
        "CITATION_NOT_ALLOWED_FOR_SECTION"
      ],
      "offending_citation_ids": [
        "E02"
      ],
      "citation_allowlist_details": [
        {
          "failure_type": "CITATION_NOT_ALLOWED_FOR_SECTION",
          "location": "sections.2.claims.2.citation_ids",
          "section_id": "results",
          "citation_ids": [
            "E02"
          ],
          "allowed_for_section": [
            "E01",
            "E03",
            "E04",
            "E06",
            "E07",
            "E08",
            "E12",
            "E13"
          ]
        }
      ],
      "research_synthesis_schema": "failed",
      "citation_allowlist": "failed"
    }
  ],
  "all_passed": false
}
```
