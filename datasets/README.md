# Datasets

The datasets are split into two folders, each with a train/test split and the full dataset:
- `balanced` contains data with an equal number of "benign" and "jailbreak" classifications
- `default` contains a standard, unmodified class balance (draws uniformly from each data source)

<br>

Each dataset contains two columns:
- `prompt`: the LLM prompt text
- `type`: "benign" (non-jailbreak) or "jailbreak"