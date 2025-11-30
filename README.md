# AI Model To Predict Water Quality

Author: Suhani Rai

 ## Overview

This repository contains Python code that:
- Loads a custom instruction dataset stored in a .json file
- Tokenizes the data using T5Tokenizer
- Fine-tunes a FLAN-T5-small model using Trainer from Hugging Face
- Saves the final fine-tuned model for downstream inference
- The project is ideal for beginners and intermediate learners experimenting with instruction tuning, LLM fine-tuning, and Hugging Face transformer workflows.

___

## Features
- Fine-tunes FLAN-T5-Small on your custom dataset
- Supports instruction → output training pairs
- Automatically tokenizes inputs and labels
- Uses Hugging Face Trainer API for easy training
- Configurable training hyperparameters
- Exports a complete fine-tuned model directory

___

## Tech Stack
- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- FLAN-T5-Small

___

## How It Works
- Load a custom dataset from mydata.json using Hugging Face Datasets
- Tokenize “instruction” and “output” fields with T5Tokenizer
- Format labels for sequence-to-sequence training
- Apply a preprocessing function across the dataset
- Create training arguments (batch size, learning rate, epochs, output directory)
- Initialize Trainer with model, dataset, and hyperparameters
- Fine-tune FLAN-T5-small on the custom data
- Save the fine-tuned model to the output directory
