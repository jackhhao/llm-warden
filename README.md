# LLM Warden

A simple jailbreak detection tool for safeguarding LLMs. Available as a fine-tuned model on HuggingFace at [jackhhao/jailbreak-classifier](https://huggingface.co/jackhhao/jailbreak-classifier).

## Description
Jailbreaking is a technique that involves creating prompts to bypass standard safety/moderation controls for LLMs. If succesful, it can lead to dangerous downstream attacks and unrestricted output. This tool serves as a way to proactively detect and defend against such attacks.

## Getting Started

### Dependencies

* Python 3

### Installation

To install, run `pip install -r requirements.txt`.


## Usage

There are three options available to start using this model:
1. Use the HuggingFace inference pipeline
2. Use the Cohere API
3. Train and run the model locally

### Using the inference pipeline
Simply run the following snippet:
```python
from transformers import pipeline

pipe = pipeline("text-classification", model="jackhhao/jailbreak-classifier")

print(pipe("is this a jailbreak?"))
```

### Using Cohere
1. Obtain a trial API key from [the Cohere dashboard](https://dashboard.cohere.com/api-keys).
2. Create a `.env` file (example one provided) with the API key.
3. Go to `cohere_client.py` and replace the classifier input with your own examples.

### Running locally
1. Run `train.py` (uses the data under `data/`).
2. Run `classify.py`, replacing the classifier input with your own examples if desired.


## Roadmap
* Create CLI tool for easy input + prediction
* Build Streamlit app to classify prompts via UI (& switch between models)
* Add moderation score / toxicity as additional model feature

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

Jack Hao - <https://www.linkedin.com/in/jackhhao>


## Acknowledgments

Thanks to the Cohere team for providing such an easy-to-use & powerful API!

And shout-out to the HuggingFace team for hosting a great platform for open-source datasets & models :)