from flair.models import TARSClassifier
from flair.data import Corpus,Sentence
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import torch
import flair
from pathlib import Path

torch.cuda.empty_cache()
import numpy as np
#from utils import stop_perf_test, start_perf_test
import time
flair.set_seed(4)
with open("snips_text.txt","r",encoding='utf-8') as f:
    text_data=f.read()
    text_data=text_data.strip()
    text_data=text_data.split("\n")

with open("snips_label.txt","r",encoding='utf-8') as f:
    label_data=f.read()
    label_data=label_data.strip()
    label_data=label_data.split("\n")

intent_list=['AddToPlaylist',
'BookRestaurant',
'GetWeather',
'PlayMusic',
'SearchScreeningEvent',
'SearchCreativeWork',
'RateBook']

index_list=[]
for label in label_data:
    for intent in intent_list:
        if label==intent:
            index_list.append(intent_list.index(intent))

def split_balanced(data, target, test_size=0.1):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])
    #print(data[list(ix_test)])
    X_test=[]
    y_test=[]
    X_train=[]
    y_train=[]
    for val in list(ix_test):
        X_test.append(data[val])
        y_test.append(target[val])
    for val in list(ix_train):
        X_train.append(data[val])
        y_train.append(target[val])
    return X_train,y_train,X_test,y_test

from datasets import  load_dataset

dataset = load_dataset("snips_built_in_intents")
train_ds=[]
test_ds=[]
train_text,train_label,test_text,test_label=split_balanced(text_data,index_list)
for datapoint,class_val in zip(train_text,train_label):
    train_ds.append(Sentence(datapoint.lower()).add_label('snips_mobile_data', intent_list[class_val].lower()))
train_ds=SentenceDataset(train_ds)

for datapoint,class_val in zip(test_text,test_label):
    test_ds.append(Sentence(datapoint.lower()).add_label('snips_mobile_data', intent_list[class_val].lower()))
test_ds=SentenceDataset(test_ds)
print (train_ds[0])
print (test_ds[0])
print("data_load Completed")

def train_module(model,fine_tune,ff_dim,nhead,epoch):
    global train_ds,test_ds
    print(f"Model : {model}\n FF_DIM : {ff_dim}\nnHead : {nhead}")

    base_path=f'taggers/snips_small_{model}_big_head_only_{ff_dim}_{nhead}'
    if type(base_path) is str:
            base_path = Path(base_path)
    corpus = Corpus(train=train_ds,test=test_ds)
    start_time= time.time()
    if model=="BERT":
        if (base_path / "checkpoint.pt").exists():
            checkpoint_model = TARSClassifier.load(base_path / "checkpoint.pt",fine_tune=fine_tune,ff_dim=ff_dim,nhead=nhead)
            checkpoint_model.add_and_switch_to_new_task("snips_mobile_data", label_dictionary=corpus.make_label_dictionary(label_type="snips_mobile_data"),label_type="snips_mobile_data")

            trainer = ModelTrainer(checkpoint_model, corpus)
            start_time= time.time()
            data= trainer.resume(model=checkpoint_model, # path to store the model artifacts
                learning_rate=0.02,
                mini_batch_size=16,
                max_epochs=epoch,
                monitor_train=False, # if we want to monitor train change to True
                embeddings_storage_mode="cuda",
                train_with_dev =True,
                checkpoint=True
                )
        else:
            print("loaded Bert")
            tars = TARSClassifier(fine_tune=fine_tune,ff_dim=ff_dim,nhead=nhead)
            tars.add_and_switch_to_new_task("snips_mobile_data", label_dictionary=corpus.make_label_dictionary(label_type="snips_mobile_data"),label_type="snips_mobile_data")

            trainer = ModelTrainer(tars, corpus)
            start_time= time.time()
            data= trainer.train(base_path=base_path, # path to store the model artifacts
                learning_rate=0.02,
                mini_batch_size=16,
                max_epochs=epoch,
                monitor_train=False, # if we want to monitor train change to True
                embeddings_storage_mode="cuda",
                train_with_dev =True,
                checkpoint=True
                )
    else:
        tars = TARSClassifier.load("tars-base",nhead=nhead,ff_dim=ff_dim,fine_tune=fine_tune)
    # start_time= time.time()

    # data= trainer.train(base_path=base_path, # path to store the model artifacts
    #             learning_rate=0.02,
    #             mini_batch_size=16,
    #             max_epochs=epoch,
    #             monitor_train=False, # if we want to monitor train change to True
    #             embeddings_storage_mode="cuda",
    #             train_with_dev =True,
    #             checkpoint=True
    #             )

    # print(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")
    print(data)

import argparse
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-m", "--model", help = "TARS/BERT", default="BERT")
    parser.add_argument("-ft", "--fine_tune", help = "Train the model (True/False)", type=bool,default=False)
    parser.add_argument("-dim", "--ffdim", help = "Feedforward Dimension Size (2048/1024/512/256)",type=int, default=2048)
    parser.add_argument("-nh", "--nhead", help = "Feedforward attention head numbers (8/4/2)", default=8,type=int)
    parser.add_argument("-e", "--epoch", help = "numbers of Epoch (50/100)", default=50,type=int)

    # Read arguments from command line
    args = parser.parse_args()

    train_module(model=args.model,fine_tune=args.fine_tune,ff_dim=args.ffdim,nhead=args.nhead,epoch=args.epoch)
