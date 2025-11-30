from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset("json", data_files="mydata.json")

def preprocess(example):
    inputs = tokenizer(example["instruction"], max_length=256, truncation=True)
    labels = tokenizer(example["output"], max_length=256, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./finetuned-flan-t5",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=3e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
