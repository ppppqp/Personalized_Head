from transformers.adapters import PrefixTuningConfig
from datasets import load_dataset
from snip_dataset import get_snip_dataset 
from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads
from transformers import BertTokenizer, BertConfig, BertModelWithHeads, BertForSequenceClassification
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from clinc_dataset import get_clinc_dataset

import numpy as np
import torch


model_name = "bert-base-uncased"
# dataset_name = "snips_built_in_intents"
dataset_name = "clinc_oos"

tokenizer = BertTokenizer.from_pretrained(model_name)

if dataset_name == "clinc_oos":
  label = "intent"
  dataset_label_num = 150
  train, test, dev = get_clinc_dataset(tokenizer = tokenizer)
else:
  label = "label"
  dataset_label_num = 7
  train, test = get_snip_dataset(tokenizer = tokenizer)

# dataset = load_dataset(dataset_name, "small")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")


# config = BertConfig.from_pretrained(
#     model_name,
#     num_labels=dataset_label_num,
# )
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels = dataset_label_num
    # config=config,
)
config = PrefixTuningConfig(flat=False, prefix_length=30)
model.add_adapter("prefix_tuning", config=config)
model.train_adapter("prefix_tuning")
# model.add_classification_head(
#     dataset_name,
#     num_labels=dataset_label_num
#   )

# print("after adaptor")
# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name, param.size())

  
training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_accuracy,
)

trainer.train()
print(trainer.evaluate())
