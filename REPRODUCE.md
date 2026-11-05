# Reproduce the checks

The archive check needs Python 3.10+ and runs on a CPU with no additional dependencies.

```sh
python experiments/check_evidence.py
```

The live experiments require a suitable PyTorch installation with CUDA, compatible GPU memory and the additional packages in `requirements.txt`. The original runtime used Transformers 5.2.0. Qwen3 checkpoints and the selected Jacobian lens are downloaded from their original sources.

```sh
python -m pip install -r requirements.txt
python experiments/run.py configs/c17_operative_8b.json runs/c17_operative_8b
```

The c17 job reruns the choice experiment with its own scene and target schedule. Its `source` field is used only by separate heading/teacher-forced subtests. Configurations that run the heading scan, “told” conditions or the *strangers* replay also refer to intermediate source records under `inputs/`. The published numerical subset does not include those full prose traces or private retrieval notes. The source manifest pins the completed records so the curated comparison can be checked separately from a live rerun. The exact model revision and accelerator image were not recorded in the retained run configurations.
