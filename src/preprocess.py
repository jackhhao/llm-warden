from datasets import (
    Dataset,
    load_dataset,
    interleave_datasets
)
from functools import partial

# NUM_SAMPLES = 7000 # if we want to control how many benign prompts we use

# add benign classification
def add_benign_label(example):
    example["type"] = "benign"
    return example

# add jailbreak classification
def add_jailbreak_label(example):
    example["type"] = "jailbreak"
    return example

# create a generator from an IterableDataset for use when converting to regular Dataset
def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

# we use OpenOrca here as an arbitrary source of benign prompts
regular_prompts = (load_dataset("Open-Orca/OpenOrca", split='train', streaming=True)#.take(50)
                    .select_columns("question")
                    .rename_column("question", "prompt")
                    .map(add_benign_label))

# add in roleplay prompts to distinguish between benign roleplays and dangerous ones (jailbreaks)
roleplay_prompts = (load_dataset('json', data_files='../data/roleplay/roleplay-instruct-v2.1.json', split='train', streaming=True)
                    .select_columns("instruction")
                    .rename_column("instruction", "prompt")
                    .map(add_benign_label))

# add in jailbreak prompts
# alternatively, can use this syntax: data_files = {"train": "train.csv", "test": "test.csv"}
jailbreak_prompts = (load_dataset("csv", data_files="../data/jailbreak/jailbreak_prompts.csv", split='train', streaming=True)
                     .select_columns("prompt")
                     .map(add_jailbreak_label))

# interleave the datasets, ensuring we have 50/50 split of benign & jailbreak
all_prompts = interleave_datasets(
    datasets=[regular_prompts, roleplay_prompts, jailbreak_prompts],
    probabilities=[0.25, 0.25, 0.5]
)

# convert IterableDataset -> Dataset
ds = Dataset.from_generator(
    partial(gen_from_iterable_dataset, all_prompts),
    features=all_prompts.features
)

# if we just want to save as CSV
ds.to_csv("../datasets/balanced/jailbreak_dataset_full_balanced.csv")

# generate train/test splits
ds = ds.train_test_split(test_size=0.2)

ds["train"].to_csv("../datasets/balanced/jailbreak_dataset_train_balanced.csv")
ds["test"].to_csv("../datasets/balanced/jailbreak_dataset_test_balanced.csv")