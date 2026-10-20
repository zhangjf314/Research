# Three real research cases

This public summary derives from the frozen PaperResearch v2 closeout. It preserves the distinctions between execution success, source auditing, human review and scientific acceptance. It contains no provider payloads or original paper assets.

| Case | Execution and report | Scientific outcome |
|---|---|---|
| COCO-LM, arXiv:2102.08473v1 | Real PDF pixels were sent through the existing Agent/Provider chain. Subsequent source audit corrected visual relations and expanded review coverage. A scoped 16-fact report was delivered in SOURCE_AUDITED_RELEASE mode. | AI source audit, human_review_completed=false, strict_trusted_publication=false. Not proof of fully correct first-pass visual understanding. |
| FNO, arXiv:2010.08895v3 | Original Runner completed a real multimodal UNVERIFIED draft. Retrieved architecture evidence was lost in assembly, and experimental/visual interpretations contained errors. | Independent project scientific acceptance **FAIL**. Execution success did not satisfy all preregistered goals. |
| U-NO, arXiv:2204.11127v3 | First request stopped at the 4096-output limit. A separately authorized 16384-output request produced complete JSON, but bare evidence aliases failed the original production citation gate. Offline normalization of 47 existing aliases rendered a separately marked derivative. | Both original run states remain INCOMPLETE; derivative UNVERIFIED; final scientific acceptance **FAIL**. No new model request was used for normalization. |

COCO-LM retains the SST-2 counterexamples: COCO-LM Base 93.30 versus RoBERTa (Ours) 93.86 and ELECTRA (Ours) 93.64. An AVG advantage was not presented as winning every task. The corrected reporting scope is the authors' own GLUE DEV runs, medians over five runs; it is not a blanket statement about every imported baseline result.

FNO's acceptance recorded A/B/G PASS, C/D/F FAIL, E/H PARTIAL. Architecture Figure 2 was retrieved but did not reach the model; Figure 3 interpretation, parameter comparisons, experimental attribution and full-prose claim coverage were not sufficiently correct.

U-NO recorded A/B PASS, C/E PARTIAL, D/F FAIL, H PARTIAL. G passed only for the explicitly derived portable artifact. The main Fourier expression was present in a truncated text window; it was not wholly absent. Main numeric results were largely traceable, but the prose mislabeled T_in as T. Whole U-shaped architecture, skip connections and key implementation context were not adequately assembled; a trend ROI did not cover the full claimed range.

The saved same-candidate comparison involved 40 ordered candidates. Candidate used 6 sources/about 8367 system+user bytes; Baseline used 13/about 14739. Both had the same 32768-byte hard limit and image/table/equation quotas. Candidate gained zero-shot Darcy evidence and the depth-memory plot, but lost architecture, the main Darcy table and some context. Both omitted an already-retrieved inference-time table. Budget was not exhausted. This is mixed evidence-selection behavior, not proof of improved research quality.

Baseline generated no comparison report. No controlled causal comparison of old/new generated report quality was performed. AI source-audit conclusions are not independent human gold.
