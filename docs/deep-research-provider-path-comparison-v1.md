# Deep Research Provider Path Comparison v1

| Item | Full QA | Deep Research |
| --- | --- | --- |
| Settings class | `paper_research.config.Settings` | `paper_research.config.Settings` |
| `.env` loading | Docker API process loads production env through Compose | Host `.venv` CLI loads project settings |
| APP_PROFILE | `production` | `production` |
| LLM_PROVIDER | `openai_compatible` | `openai_compatible` |
| LLM_PROVIDER_NAME | `deepseek` | `deepseek` |
| LLM_BASE_URL | `api.deepseek.com` | `api.deepseek.com` |
| LLM_MODEL | `deepseek-v4-flash` | `deepseek-v4-flash` |
| API key | present; fingerprint `23e1f83b`; full value not recorded | present; fingerprint `23e1f83b`; full value not recorded |
| Provider Factory function | `build_llm_provider(settings)` inside API service | `build_llm_provider(settings.model_copy(update={"llm_max_retries": 0}))` |
| Client class | `httpx.Client` through `OpenAICompatibleLLMProvider` | `httpx.Client` through `OpenAICompatibleLLMProvider` |
| sync/async | sync | sync |
| timeout | `120s` | `120s` |
| connect/read timeout | httpx scalar timeout | httpx scalar timeout |
| proxy/trust_env | httpx default | httpx default |
| SSL verify | httpx default | httpx default |
| retry policy | QA runner configured no QA retry for final route; provider config persisted | Deep Research smoke forces provider `llm_max_retries=0` |
| extra_body | DeepSeek `thinking.type=disabled` | DeepSeek `thinking.type=disabled` |
| response_format | `json_object` | `json_object` |
| thinking | disabled | disabled |
| execution process | Docker API container via `http://localhost/api/v1` | Host `.venv` CLI process |
| host/container | Container performs provider POST | Host process performs provider POST |

Conclusion: Full QA and Deep Research now resolve the same configured DeepSeek provider through the shared provider factory. The observed difference is execution environment: Full QA provider calls were made inside the Docker API container and completed 50/50, while the Deep Research host CLI POST reproduced a socket permission failure in the normal Codex execution context.
