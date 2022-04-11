from transformers import AutoTokenizer
import numpy as np
from torch import nn
import torch
from datasets import load_dataset
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from clinc_dataset import get_clinc_dataset
from transformers import EvalPrediction
from torchsummary import summary

# output weights: https://discuss.pytorch.org/t/how-to-output-weight/2796 
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True, num_labels=150)

model.to("cuda")

train, test, dev = get_clinc_dataset(tokenizer = tokenizer)
print("#################BEFORE TRAINIG##################")
for name, param in model.named_parameters():
    # print(name, param.data)
    if 'classifier' not in name: # classifier layer
        param.requires_grad = False



def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    if type(output) == torch.Tensor:
        print("norm:", output.data.norm())

for module in model.modules():
    if not isinstance(module, nn.Sequential):
        module.register_forward_hook(printnorm)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train,         # training dataset
    eval_dataset=test,          # evaluation dataset
    compute_metrics=compute_accuracy,     # the callback that computes metrics of interest
)

trainer.train()

print("#################AFTER TRAINIG##################")
# for name, param in model.named_parameters():
    # print(name, param.data)

print(trainer.evaluate())