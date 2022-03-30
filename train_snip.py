import torch
from datasets import  load_dataset
from torch.utils.data import Dataset
import torch.nn.functional as F
torch.cuda.empty_cache()
import numpy as np
import time




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





def get_snip_data():
    with open("snips/snips_text.txt","r",encoding='utf-8') as f:
        text_data=f.read()
        text_data=text_data.strip()
        text_data=text_data.split("\n")

    with open("snips/snips_label.txt","r",encoding='utf-8') as f:
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
    train_ds=[]
    test_ds=[]
    train_text,train_label,test_text,test_label=split_balanced(text_data,index_list)
    new_label=[]
    new_text = []
    count=0
    for i in range(0,len(train_label),1861):
        if count< 7:
            for j in range(i,i+19):
                new_label.append(train_label[j])
                new_text.append(train_text[j])
            count+=1
    [new_label.append(label) for label in train_label[-7:]]
    [new_text.append(text) for text in train_text[-7:]]
    return new_text, new_label, test_text, test_label

def get_snip_dataset(tokenizer):
    train_text, train_label, test_text, test_label = get_snip_data()
    return SnipDataset(train_text, train_label, tokenizer), SnipDataset(test_text, test_label, tokenizer)



class SnipDataset(Dataset):
    def __init__(self, text, label, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        # for i in range(len(text)):
        #     self.text.append(tokenizer(text[i], max_length=80, truncation=True, padding="max_length"))
        # self.label = F.one_hot(torch.tensor(label), num_classes=7)
        self.label = label
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        item = self.tokenizer(self.text[idx], max_length=80, truncation=True, padding="max_length")
        item["label"] = self.label[idx]
        return item


def train_module(model,fine_tune,ff_dim,nhead):
    global train_ds,test_ds

    corpus = Corpus(train=train_ds,test=test_ds)
    start_time= time.time()
    if model=="BERT":
        tars = TARSClassifier(fine_tune=fine_tune,ff_dim=ff_dim,nhead=nhead)
    else:
        tars = TARSClassifier.load("tars-base")
    print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")

    tars.add_and_switch_to_new_task("snips_mobile_data", label_dictionary=corpus.make_label_dictionary(label_type="snips_mobile_data"),label_type="snips_mobile_data")

    trainer = ModelTrainer(tars, corpus)

    start_time= time.time()

    data= trainer.train(base_path=f'taggers/clinc_small_tars_big_head_only_{ff_dim}_{nhead}', # path to store the model artifacts
                learning_rate=0.02,
                mini_batch_size=16,
                max_epochs=50,
                monitor_train=False, # if we want to monitor train change to True
                embeddings_storage_mode="cuda",
                train_with_dev =True
                )

    print(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")
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

    # Read arguments from command line
    args = parser.parse_args()
    #print(args.model,args.fine_tune,args.ffdim,args.nhead)
    train_module(args.model,args.fine_tune,args.ffdim,args.nhead)

# corpus = Corpus(train=train_ds,test=test_ds)
# start_time= time.time()
# # 1. load base TARS
# tars = TARSClassifier()#.load("tars-base")
# # print(tars)
# print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")
# # 2. make the model aware of the desired set of labels from the new corpus
# tars.add_and_switch_to_new_task("snips_mobile_data", label_dictionary=corpus.make_label_dictionary(label_type="snips_mobile_data"),label_type="snips_mobile_data")
# # 3. initialize the text classifier trainer with your corpus
# start_time= time.time()

# trainer = ModelTrainer(tars, corpus)
# # start_perf_test()
# # 4. train model
# data=trainer.train(base_path='taggers/snips_full_small_mobile_with_big_head_2', # path to store the model artifact
#               learning_rate=0.02, 
#               mini_batch_size=16, 
#               max_epochs=2,
#               shuffle=True,
#               monitor_train=False,
#               train_with_dev =True,
#               embeddings_storage_mode="cuda")
# # stop_perf_test()
# print(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")
# print(data)