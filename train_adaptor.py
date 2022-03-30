from datasets import load_dataset
from train_snip import get_snip_dataset 

from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads
from transformers import BertTokenizer, BertConfig, BertModelWithHeads
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
import numpy as np
import torch
from torchsummary import summary
# reference: https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
# python 3.8 cuda 11.2 install using: pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

model_name = "bert-base-uncased"
dataset_name = "snips_built_in_intents"
# dataset_name = "clinc_oos"
dataset_label_num = 150

if dataset_name == "clinc_oos":
  label = "intent"
else:
  label = "label"

dataset = load_dataset(dataset_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
print(dataset)
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"

dataset = dataset.rename_column(label, "labels")
# print(dataset['train'][0])
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

if dataset_name == "snips_built_in_intents":
  dataset = dataset["train"].train_test_split(test_size=0.2)


print(dataset['train'][0])
config = BertConfig.from_pretrained(
    model_name,
    num_labels=dataset_label_num,
)
model = BertModelWithHeads.from_pretrained(
    model_name,
    config=config,
)

model.add_adapter(dataset_name)
model.train_adapter(dataset_name)
model.add_classification_head(
    dataset_name,
    num_labels=dataset_label_num
  )

print("after adaptor")
for name, param in model.named_parameters():
  if param.requires_grad:
    print(name, param.size())

  
training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=10,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

train, test = get_snip_dataset(tokenizer = tokenizer)
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=compute_accuracy,
)

trainer.train()
print(trainer.evaluate())