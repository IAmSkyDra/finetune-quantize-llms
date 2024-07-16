## How to fine-tune and quantize an LLM with Google Colab easily?

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab (vi)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17IDka1ZMj6Hw5WmuwiYaHGBQ-bnnX3hb?usp=sharing)

### Introduction

This repository contains materials for the CSE Summer School Hackathon 2024, aimed at guiding **high school students** on how to fine-tune and quantize large language models (LLMs) with Google Colab in the simplest way possible. It **minimizes the need for coding** and requires only minor adjustments to pre-designed functions.

Key information:
* Fine-tuning the model with [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).
* Quantizing the model with [Unsloth](https://github.com/unslothai/unsloth).
* The model being fine-tuned in the notebook is [GemSUra 2B](https://huggingface.co/ura-hcmut/GemSUra-2B).
* The type of fine-tuning fixed in the notebook is LoRA using SFT with sample data from [HCMUT_FAQ](https://huggingface.co/datasets/IAmSkyDra/HCMUT_FAQ). Students are guided on how to create their own data based on the competition theme.
* The quantization method in the notebook is 8-bit, enabling students to deploy the model on personal computers using [Ollama](https://github.com/ollama/ollama).
  
### Notebook usage guide

#### Install required packages

```python
!pip install --upgrade -r requirements.txt -q

from module import *
```

#### Log in to Hugging Face

```python
hf("HF_TOKEN")
```

#### Load dataset

```python
hcmut_dataset = load_dataset("IAmSkyDra/HCMUT_FAQ", split="train", streaming=False)

identity_dataset = load_dataset("IAmSkyDra/HCMUT_FAQ", split="validation", streaming=False)
```

#### Train the model

```python
train([identity_dataset, hcmut_dataset], num_train_epochs, continue_training)
```

#### Evaluate the fine-tuned model

```python
test()
```

#### Merge LoRA adapters with the model and upload to Hugging Face

```python
merge_and_push("IAmSkyDra/GemSUra-edu")
```

#### Model inference

```python
inference("IAmSkyDra/GemSUra-edu")

chat(max_new_tokens, history)
```

#### Quantize the model and upload to Hugging Face

```python
quantize_and_push("IAmSkyDra/GemSUra-edu-quantized")
```

> Detailed explanations of each function and their arguments, as well as other information, are provided in the notebook.

### Conclusion

By the end of this notebook, students will have a fine-tuned model tailored with custom data based on the pre-trained GemSUra 2B model, along with its quantized version.
