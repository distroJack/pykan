import sys

import numpy as np


class EarlyStopping:
    def __init__(self, train_patience=5, val_patience=5, verbose=False, delta=0):
        """
        Args:
            train_patience (int): How long to wait after last time train loss improved.
                            Default: 5
            val_patience (int): How long to wait after last time val loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.train_patience = train_patience
        self.val_patience = val_patience
        self.verbose = verbose
        self.delta = delta

        self.train_counter = 0
        self.val_counter = 0

        self.best_train_loss = None
        self.best_val_loss = None

        self.train_loss_min = np.Inf
        self.val_loss_min = np.Inf

        self.early_stop = False

    def __call__(self, train_loss, val_loss):
        if self.best_train_loss is None or self.best_val_loss is None:
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.train_loss_min = train_loss
            self.val_loss_min = val_loss

        if val_loss > self.best_val_loss - self.delta:
            self.val_counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping val counter: {self.val_counter} out of {self.val_patience}"
                )
                sys.stdout.flush()
            if self.val_counter >= self.val_patience:
                self.early_stop = True
                return
        else:
            self.best_val_loss = val_loss
            self.val_loss_min = val_loss
            self.val_counter = 0

        if train_loss > self.best_train_loss - self.delta:
            self.train_counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping train counter: {self.train_counter} out of {self.train_patience}"
                )
                sys.stdout.flush()
            if self.train_counter >= self.train_patience:
                self.early_stop = True
        else:
            self.best_train_loss = train_loss
            self.train_loss_min = train_loss
            self.train_counter = 0
