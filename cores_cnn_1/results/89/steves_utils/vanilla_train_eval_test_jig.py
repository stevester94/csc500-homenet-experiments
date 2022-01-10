#! /usr/bin/env python3
from numpy.lib.utils import source
import torch.nn as nn
import time
import torch.optim as optim
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from steves_utils.utils_v2 import do_graph


class Vanilla_Train_Eval_Test_Jig:
    def __init__(
        self,
        model:nn.Module,
        label_loss_object,
        path_to_best_model:str,
        device,
    ) -> None:
        self.model = model.to(device)
        self.label_loss_object = label_loss_object.to(device)
        self.path_to_best_model = path_to_best_model
        self.device = device


    def train(self,
        train_iterable,
        source_val_iterable,
        target_val_iterable,
        num_epochs:int,
        num_logs_per_epoch:int,
        patience:int,
        criteria_for_best:str, # "source", "target", "source_and_target"
    ):
        last_time = time.time()

        num_batches_per_epoch = len(train_iterable)

        batches_to_log = np.linspace(1, num_batches_per_epoch, num=num_logs_per_epoch, endpoint=False).astype(int)

        for p in self.model.parameters():
            p.requires_grad = True

        history = {}
        history["epoch_indices"] = []
        history["train_label_loss"] = []
        history["source_val_label_loss"] = []
        history["target_val_label_loss"] = []

        best_epoch_index_and_loss = [0, float("inf")]
        for epoch in range(1,num_epochs+1):
            train_iter = iter(train_iterable)
            
            train_label_loss_epoch = 0
            num_examples_processed = 0

            for i in range(num_batches_per_epoch):
                self.model.zero_grad()

                """
                Do forward on source
                """
                x,y = train_iter.next()
                num_examples_processed += x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
 

                learn_results = self.model.learn(x, y)
                batch_label_loss = learn_results["label_loss"]

                train_label_loss_epoch += batch_label_loss.cpu().item()

                if i in batches_to_log:
                    cur_time = time.time()
                    examples_per_second =  num_examples_processed / (cur_time - last_time)
                    num_examples_processed = 0
                    last_time = cur_time
                    sys.stdout.write(
                        (
                            "epoch: {epoch}, [batch: {batch} / {total_batches}], "
                            "examples_per_second: {examples_per_second:.4f}, "
                            "train_label_loss: {train_label_loss:.4f}, "
                            "\n"
                        ).format(
                                examples_per_second=examples_per_second,
                                epoch=epoch,
                                batch=i,
                                total_batches=num_batches_per_epoch,
                                train_label_loss=batch_label_loss.cpu().item(),
                            )
                    )

                    sys.stdout.flush()

            source_val_acc_label, source_val_label_loss = self.test(source_val_iterable)
            target_val_acc_label, target_val_label_loss = self.test(target_val_iterable)

            history["epoch_indices"].append(epoch)
            history["train_label_loss"].append(train_label_loss_epoch / num_batches_per_epoch)
            history["source_val_label_loss"].append(source_val_label_loss)
            history["target_val_label_loss"].append(target_val_label_loss)

            sys.stdout.write(
                (
                    "=============================================================\n"
                    "epoch: {epoch}, "
                    "source_val_acc_label: {source_val_acc_label:.4f}, "
                    "source_val_label_loss: {source_val_label_loss:.4f}, "
                    "target_val_acc_label: {target_val_acc_label:.4f}, "
                    "target_val_label_loss: {target_val_label_loss:.4f}, "
                    "\n"
                    "=============================================================\n"
                ).format(
                        epoch=epoch,
                        source_val_acc_label=source_val_acc_label,
                        source_val_label_loss=source_val_label_loss,
                        target_val_acc_label=target_val_acc_label,
                        target_val_label_loss=target_val_label_loss,
                    )
            )

            sys.stdout.flush()

            # New best, save model
            if criteria_for_best == "source": criteria_loss = source_val_label_loss
            elif criteria_for_best == "target": criteria_loss = target_val_label_loss
            elif criteria_for_best == "source_and_target": criteria_loss = source_val_label_loss + target_val_label_loss
            else: raise ValueError("criteria for best is not valid")

            if best_epoch_index_and_loss[1] > criteria_loss:
                print("New best")
                best_epoch_index_and_loss[0] = epoch
                best_epoch_index_and_loss[1] = criteria_loss
                torch.save(self.model.state_dict(), self.path_to_best_model)
            
            # Exhausted patience
            elif epoch - best_epoch_index_and_loss[0] > patience:
                print("Patience ({}) exhausted".format(patience))
                break
        
        self.model.load_state_dict(torch.load(self.path_to_best_model))
        self.history = history

    def test(self, iterable):
        with torch.no_grad():
            n_batches = 0
            n_total = 0
            n_correct = 0

            total_label_loss = 0

            model = self.model
            model.eval()

            for x,y in iter(iterable):
                batch_size = len(x)

                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = model.forward(x) # Forward does not use alpha
                pred = y_hat.data.max(1, keepdim=True)[1]

                n_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
                n_total += batch_size

                total_label_loss += self.label_loss_object(y_hat, y).cpu().item()

                n_batches += 1

            accu = n_correct.data.numpy() * 1.0 / n_total
            average_label_loss = total_label_loss / n_batches

            model.train()

            return accu, average_label_loss
    
    def get_history(self):
        return self.history

    @classmethod
    def do_diagram(cls, history, axis):
        """
        returns: Writes to axis [
            [loss curve]
        ]
        """
      
        # Bottom Right: src train label vs  src val label
        graphs = [
            {
                "x": history["epoch_indices"],
                "y": history["train_label_loss"],
                "x_label": None,
                "y_label": "Train Label Loss",
                "x_units": "Epoch",
                "y_units": None,
            }, 
            {
                "x": history["epoch_indices"],
                "y": history["source_val_label_loss"],
                "x_label": None,
                "y_label": "Source Val Label Loss",
                "x_units": "Epoch",
                "y_units": None,
            },
            {
                "x": history["epoch_indices"],
                "y": history["target_val_label_loss"],
                "x_label": None,
                "y_label": "Target Val Label Loss",
                "x_units": "Epoch",
                "y_units": None,
            }, 
        ]
        do_graph(axis, "Source Train Label Loss vs Source Val Label Loss", graphs)