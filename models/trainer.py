import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import numpy as np
import argparse
from rdkit import Chem

import time
import csv
import os

# import modeuls from current directory
from dataset import *
import model as m
from callback import *


class Trainer(object):
    def __init__(self, dataset, split_type):
        self.data = None
        self.model = None
        self.hyper = {"dataset": dataset}
        self.split = split_type
        self.log = {}

    def __repr__(self):
        text = ""
        for key, value in self.log.items():
            text += "{}:\t".format(key)
            for error in value[0]:
                text += "{0:.4f} ".format(float(error))
            text += "\n"

        return text

    def load_data(self, iter, batch=128, fold=10):
        self.data = Dataset(self.hyper["dataset"], batch=batch, fold=fold, iter=iter, split=self.split)
        self.hyper["num_train"] = len(self.data.y["train"])
        self.hyper["num_test"] = len(self.data.y["test"])
        self.hyper["num_atoms"] = self.data.max_atoms
        self.hyper["num_features"] = self.data.num_features
        self.hyper["data_std"] = self.data.std
        self.hyper["data_mean"] = self.data.mean
        self.hyper["task"] = self.data.task
        self.hyper["outputs"] = self.data.outputs
        self.hyper["batch"] = batch

    def load_model(self, model, units_conv=128, units_dense=128, num_layers=2, loss="mse", pooling="max"):
        self.hyper["model"] = model
        self.hyper["units_conv"] = units_conv
        self.hyper["units_dense"] = units_dense
        self.hyper["num_layers"] = num_layers
        self.hyper["loss"] = loss
        self.hyper["pooling"] = pooling
        self.model = getattr(m, model)(self.hyper)
        # self.model.summary()

    def fit(self, model, epoch, batch=128, fold=10, pooling="max", units_conv=128, units_dense=128, num_layers=2,
            loss="mse", monitor="val_rmse", mode="min", use_multiprocessing=False, label="", **kwargs):

        # 1. Generate CV folder
        now = datetime.now()
        base_path = "./result/{}/{}/".format(model, self.hyper["dataset"])
        if not (os.path.isdir(base_path)):
            os.mkdir(base_path)

        log_path = base_path + "{}_c{}_d{}_l{}_p{}_{}{}/".format(batch, units_conv, units_dense, num_layers,
                                                                 pooling, label, now.strftime("%m%d%H"))
        if not (os.path.isdir(log_path)):
            os.mkdir(log_path)
        print('save path: ', log_path)

        results = []
        for i in range(fold):
            start_time = time.time()

            # 2. Generate data
            self.load_data(batch=batch, fold=fold, iter=i)
            self.data.set_features(**kwargs)
            self.hyper["num_features"] = self.data.num_features

            # 3. Make model
            self.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers, loss=loss,
                            pooling=pooling)

            # 4. Callbacks
            tb_path = log_path + "trial_{}/".format(i)
            if not (os.path.isdir(tb_path)):
                os.mkdir(tb_path)

            callbacks = []
            if self.data.task != "regression":
                callbacks.append(Roc(self.data.generator("test")))
                mode = "max"
            callbacks += [Tensorboard(log_dir=tb_path, write_graph=False, histogram_freq=0, write_images=True),
                          ModelCheckpoint(tb_path + "{epoch:01d}-{" + monitor + ":.3f}.hdf5", monitor=monitor,
                                          save_weights_only=True, save_best_only=True, period=1, mode=mode),
                          EarlyStopping(patience=15, restore_best_weights=True),  # 15, hiv=10
                          ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=10, min_lr=0.0005)]

            # 5. Fit
            self.model.fit_generator(self.data.generator("train"), epochs=epoch,
                                     validation_data=self.data.generator("test"), callbacks=callbacks)
            self.model.save_weights(tb_path + "best_weight.hdf5")
            self.hyper["train_time"] = time.time() - start_time

            # 6. Save train, test losses
            if self.data.task == "regression":
                train_loss = self.model.evaluate_generator(self.data.generator("train"))
                test_loss = self.model.evaluate_generator(self.data.generator("test"))

                results.append([train_loss[1], test_loss[1], train_loss[2], test_loss[2]])

            else:
                losses = []
                for gen in [self.data.generator("train"), self.data.generator("test")]:
                    val_roc, val_pr, val_f1 = calculate_roc_pr_f1(self.model, gen)
                    losses.append(val_roc)
                    losses.append(val_pr)
                    losses.append(val_f1)

                results.append([losses[0], losses[3], losses[1], losses[4], losses[2], losses[5]])

            # 7. Save hyper
            with open(tb_path + "hyper.csv", "w") as file:
                writer = csv.DictWriter(file, fieldnames=list(self.hyper.keys()))
                writer.writeheader()
                writer.writerow(self.hyper)

            # 8. Save data split and test results
            for target in ["train", "test"]:
                pred = self.model.predict_generator(self.data.generator(target, task="input_only"))
                self.data.save_dataset(tb_path, pred=pred, target=target)

        # 9. Save cross validation results
        if self.data.task == "regression":
            header = ["train_mae", "test_mae", "train_rmse", "test_rmse"]
        else:
            header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1"]

        with open(log_path + "raw_results.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)

        results = np.array(results)
        results = [np.mean(results, axis=0), np.std(results, axis=0)]
        with open(log_path + "k_fold_results.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)

        print(results)

        # Update cross validation log
        self.log["{}_B{}_C{}_D{}_L{}_P{}".format(model, batch, units_conv, units_dense, num_layers, pooling,
                                                 )] = results

        print(self)
        print("Training Ended")


def load_model(trial_path):
    with open(trial_path + "/hyper.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            hyper = dict(row)

    dataset = hyper['dataset']
    model = hyper['model']
    batch = int(hyper['batch'])
    units_conv = int(hyper['units_conv'])
    units_dense = int(hyper['units_dense'])
    num_layers = int(hyper['num_layers'])
    loss = hyper['loss']
    pooling = hyper['pooling']
    std = float(hyper['data_std'])
    mean = float(hyper['data_mean'])

    # Load model
    trainer = Trainer(dataset, split_type="stratified_sampling")
    trainer.load_data(batch=batch, iter=1)
    trainer.data.std = std
    trainer.data.mean = mean
    trainer.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers,
                       loss=loss, pooling=pooling)

    # Load best weight
    trainer.model.load_weights(trial_path + "/best_weight.hdf5")
    print("Loaded Weights from {}".format(trial_path + "/best_weight.hdf5"))

    return trainer, hyper


def model_evaluate(path):
    # Load the trained model
    fold = ['trial_' in filename for filename in os.listdir(path)].count(True)
    results = []

    for i in range(0, fold):
        trial_path = path + "/trial_" + str(i)
        trainer, hyper = load_model(trial_path)

        # Load existed dataset
        trainer.data.replace_dataset(trial_path + "/train.sdf", subset="train", target_name="true")
        trainer.data.replace_dataset(trial_path + "/test.sdf", subset="test", target_name="true")

        # Save results
        if hyper["task"] == "regression":
            train_loss = trainer.model.evaluate_generator(trainer.data.generator("train"))
            test_loss = trainer.model.evaluate_generator(trainer.data.generator("test"))
            results.append([train_loss[1], test_loss[1], train_loss[2], test_loss[2]])

        else:
            train_roc, train_pr, train_f1 = calculate_roc_pr_f1(trainer.model, trainer.data.generator("train"))
            val_roc, val_pr, val_f1 = calculate_roc_pr_f1(trainer.model, trainer.data.generator("test"))
            results.append([train_roc, val_roc, train_pr, val_pr, train_f1, val_f1])

    # Save cross validation results
    if hyper["task"] == "regression":
        header = ["train_mae", "test_mae", "train_rmse", "test_rmse"]
    else:
        header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1"]

    with open(path + "/eval_raw_results.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        for r in results:
            writer.writerow(r)

    results = np.array(results)
    results = [np.mean(results, axis=0), np.std(results, axis=0)]
    with open(path + "/eval_results.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(header)
        for r in results:
            writer.writerow(r)

    print(results)
    print("Evaluation Ended")


