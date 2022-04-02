from flair.models import TARSClassifier
from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import time
import pickle
from pathlib import Path
import argparse
train_ds = []
test_ds = []
dev_ds = []


def load_data():
    global train_ds, test_ds, dev_ds
    file = open('train_data_clinc_wo_oos.pkl', 'rb')
    train = pickle.load(file)
    print("Loaded Train Data")
    train_ds = SentenceDataset(train)
    print(train_ds[0])

    file = open('test_data_clinc_wo_oos.pkl', 'rb')
    test = pickle.load(file)
    print("Loaded Test Data")
    test_ds = SentenceDataset(test)
    print(test_ds[0])

    file = open('dev_data_clinc_wo_oos.pkl', 'rb')
    dev = pickle.load(file)
    print("Loaded Dev Data")
    dev_ds = SentenceDataset(dev)
    print(dev_ds[0])
    print("Data loading completed")


def train_module(model, fine_tune, ff_dim, nhead):
    global train_ds, test_ds, dev_ds
    load_data()
    base_path = f'taggers/clinc_plus_{model}_with_head_only_{ff_dim}_{nhead}'
    if type(base_path) is str:
        base_path = Path(base_path)
    corpus = Corpus(train=train_ds, test=test_ds, dev=dev_ds)
    start_time = time.time()
    if model == "BERT":
        if (base_path / "checkpoint.pt").exists():
            checkpoint_model = TARSClassifier.load(
                base_path / "checkpoint.pt", fine_tune=fine_tune, ff_dim=ff_dim, nhead=nhead)
            checkpoint_model.add_and_switch_to_new_task("clinc_data", label_dictionary=corpus.make_label_dictionary(
                label_type="clinc_data"), label_type="clinc_data")

            trainer = ModelTrainer(checkpoint_model, corpus)
            start_time = time.time()
            data = trainer.resume(model=checkpoint_model,  # path to store the model artifacts
                                  learning_rate=0.02,
                                  mini_batch_size=16,
                                  max_epochs=100,
                                  monitor_train=False,  # if we want to monitor train change to True
                                  embeddings_storage_mode="cuda",
                                  train_with_dev=True,
                                  checkpoint=True
                                  )
            return
        else:
            tars = TARSClassifier(fine_tune=fine_tune,
                                  ff_dim=ff_dim, nhead=nhead)
    else:
        tars = TARSClassifier.load(
            "tars-base", fine_tune=fine_tune, ff_dim=ff_dim, nhead=nhead)
    print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")

    tars.add_and_switch_to_new_task("clinc_data", label_dictionary=corpus.make_label_dictionary(
        label_type="clinc_data"), label_type="clinc_data")

    trainer = ModelTrainer(tars, corpus)

    start_time = time.time()

    data = trainer.train(base_path=f'taggers/clinc_plus_{model}_with_head_only_{ff_dim}_{nhead}',  # path to store the model artifacts
                         learning_rate=0.02,
                         mini_batch_size=16,
                         max_epochs=1,
                         monitor_train=False,  # if we want to monitor train change to True
                         embeddings_storage_mode="cuda",
                         train_with_dev=True,
                         checkpoint=True
                         )

    print(
        f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")
    print(data)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-m", "--model", help="TARS/BERT", default="BERT")
    parser.add_argument(
        "-ft", "--fine_tune", help="Train the model (True/False)", type=bool, default=False)
    parser.add_argument(
        "-dim", "--ffdim", help="Feedforward Dimension Size (2048/1024/512/256)", type=int, default=2048)
    parser.add_argument(
        "-nh", "--nhead", help="Feedforward attention head numbers (8/4/2)", default=8, type=int)

    # Read arguments from command line
    args = parser.parse_args()
    # print(args.model,args.fine_tune,args.ffdim,args.nhead)
    train_module(args.model, args.fine_tune, args.ffdim, args.nhead)
