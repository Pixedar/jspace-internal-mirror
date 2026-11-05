# Evidence and file guide

The release preserves trial-level and token-level measurements from selected 4–5 November runs. The complete decision rows for the selected conditions are included. Generated passages, thinking traces, prompts and checkpoints are outside this numerical subset. The scene table preserves scores and binary word-match observations produced by applying the original analysis's stem rule to the saved passages. All tables use UTF-8 CSV. Blank values are inapplicable or unavailable.

| File | Unit of observation | What to look for |
|---|---|---|
| `scene_transitions.csv` | One previous readout entry per transition, frame and life | `word_source` distinguishes the run's own readout from the list actually shown. `written` flags a stem match in the next passage and `next_readout` flags exact reappearance in the next top six. Use the own rows for the across-frame comparison, including the no-readout control. |
| `heldout_heads.csv` | One held-out context per intervention | Target-word log probabilities. The baseline has all three framings. Other rows give the own and reworded-own scores after blocking attention to the heading. |
| `decision_rounds.csv` | One decision round per condition, run and life | A means own workspace. B means plain words. Readiness is a rescaled measured target log probability. `p_a` is choice probability, and forecast columns contain the model's separately elicited scores. `forced_a` identifies system choices in the 40-round test. `intervention_active` marks the hidden intervention; `selfcorrect_protocol` distinguishes a route cut from activation amplification. |
| `strangers_ranks.csv` | One reconstruction condition | Rank in the 20,000-word lens vocabulary. The output-layer rank uses a different vocabulary. |
| `strangers_regenerations.csv` | One regeneration from the selected trajectory | `units` records whether the added concept units were active. The displayed readout position and the larger vocabulary rank use different baselines. |
| `strangers_geometry.csv` | One concept comparison | Cosine similarity in the lens geometry. |
| `source-records.json` | One source run | Original saved-result basename and SHA-256, with the job configuration. File references were normalized to relative paths. |

A life is a run within a condition. Pair by run and life index when comparisons share starting scenes or seeds. Treat rounds, words and regenerated replies from one trajectory as dependent. For scene uptake, average within each life before averaging lives. For the choice result, count a life when B was chosen in more than half its STRONG rounds in the second half. The 8B heading scan, loop, and invalid first score calibration share run c13. The decision table starts from the corrected c16 run.

Run `python experiments/check_evidence.py` from the repository root for the headline checks. The original analyses and full experiment drivers are in [experiments](experiments/). The same numerical subset is mirrored in the [Hugging Face dataset](https://huggingface.co/datasets/Pixedar/jspace-internal-mirror-evidence).

The companion dataset DOI is [10.5281/zenodo.23989888](https://doi.org/10.5281/zenodo.23989888). It was reserved for this dataset. The public DOI link becomes usable after the deposit is published.
