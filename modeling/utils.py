import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, recall_score,precision_score,roc_auc_score
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, recall_score,precision_score
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score, AUROC, Accuracy
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


import numpy as np

def plot_val_train_loss(history):

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_score'], label='Train F1 Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.show()

def get_metrics(model, X_test, y_test, threshold):
    preds = np.where(model.predict(X_test)>threshold, 1, 0)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1 Score': f1_score(y_test, preds),
        'AUC': roc_auc_score(y_test, model.predict(X_test)),
    }
    return metrics

def get_metrics_pytorch(preds, y_test):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1 Score': f1_score(y_test, preds),
    }
    return metrics


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def predict(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs[0])
            predicted_labels = (outputs > 0.5).float()  # Convert probabilities to binary output
            predictions.append(predicted_labels)
    return torch.cat(predictions).numpy()  # Return predictions as a numpy array

def train_model(model, epochs, train_loader, val_loader, optimizer, loss_fn, accuracy_metric, auc_metric, f1_score_metric):
    train_losses = []
    val_losses = []

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = loss_fn(outputs, labels)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
    return train_losses, val_losses


def train_model_early_stopping(model, epochs, train_loader, val_loader, optimizer, loss_fn, accuracy_metric, auc_metric, f1_score_metric, early_stopping):
    train_losses = []
    val_losses = []
    best_f1_score = 0  # Initialize the best F1 Score for early stopping

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_f1_score = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = loss_fn(outputs, labels)
                total_val_loss += val_loss.item()

                # Update metrics
                accuracy_metric(outputs, labels.int())
                auc_metric(outputs, labels.int())
                f1_score_metric(outputs, labels.int())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Calculate metrics
            val_accuracy = accuracy_metric.compute()
            val_auc = auc_metric.compute()
            val_f1_score = f1_score_metric.compute()

            # Reset metrics for the next epoch
            accuracy_metric.reset()
            auc_metric.reset()
            f1_score_metric.reset()

            print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}, "
                  f"Validation Loss = {avg_val_loss:.4f}, "
                  f"Accuracy = {val_accuracy:.4f}, AUC = {val_auc:.4f}, F1 Score = {val_f1_score:.4f}")

        # Early stopping logic
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
            early_stopping.counter = 0  # Reset counter if there is improvement
        else:
            early_stopping.counter += 1  # Increment counter if no improvement

        if early_stopping.counter > early_stopping.patience:
            print("Early stopping")
            break  # Break out of the loop if early stopping is triggered

    return train_losses, val_losses


def plot_roc_curves(X_test, y_test, models):
    plt.figure(figsize=(10, 8))
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Colors for the lines
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    # Check if we have more models than colors and reuse colors if necessary
    if len(models) > len(colors):
        colors = colors * (len(models) // len(colors) + 1)
    
    # Iterate through the list of models
    for i, model in enumerate(models):
        probs = model.predict(X_test)
        preds = probs
        fpr, tpr, _ = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], label='Model {} AUC = {:.4f}'.format(i+1, roc_auc))
    
    plt.legend(loc='lower right')
    plt.show()