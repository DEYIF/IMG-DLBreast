#!/usr/bin/env python3

# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

#!/usr/bin/env python3

# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch
import numpy as np
from skimage import io
from sklearn import metrics

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader
import torchvision.models as models
import argparse
import tqdm

#add path
sys.path.append('/content/drive/MyDrive')

from dataImport import DBTData
from utils import accuracy
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import classification_report, f1_score, recall_score
from torchvision.models import shufflenet_v2_x1_0
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tqdm
import numpy as np
import seaborn as sns

# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# properties
batch_size = 10
val_batch_size = 1
num_classes = 2
num_mc = 50

# load data
color = True
resize_to = (256, 256)
color = True
resize_to = (256, 256)
snapshot="/content/drive/MyDrive/snapshots_best.pth.tar"
dataset_test = DBTData('train_phase2',"valid",crop_to=(224,224), resize_to=resize_to, color=color)
dataloader_test = DataLoader(dataset_test, batch_size=val_batch_size, shuffle=False)

assert len(dataset_test) > 0

print("Test dataset length:", len(dataset_test))
print('')

# create a model
model = torch.nn.Module()
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# load weights for flow estimation from best last stage
checkpoint = torch.load(snapshot, map_location=device)
print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from " + snapshot)
model.load_state_dict(checkpoint['state_dict'])

# Create loss function
criterion = torch.nn.CrossEntropyLoss()

# Go through test set
print("Going through test set.")
model.eval()
y_true_np = []
y_pred_np = []
y_pred_prob_np = []

with torch.no_grad():
    test_losses = []
    test_accuracies = []
    correct_var = []
    incorrect_var = []

    batches = tqdm.tqdm(dataloader_test)
    for i, (x, y) in enumerate(batches):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        # Use softmax to get the predicted probabilities
        y_pred_prob = F.softmax(y_pred, dim=1)

        y_true_np.append(y.data.cpu().numpy())
        y_pred_np.append(y_pred_prob.argmax(dim=1).data.cpu().numpy())
        y_pred_prob_np.append(y_pred_prob.data.cpu().numpy())

        y = y.to(torch.long)
        mean = y_pred_prob.to(torch.float32)

        test_loss = criterion(mean, y)
        test_losses.append(test_loss.item())

print("test mean loss:", np.mean(test_losses))

y_pred_np = np.array(y_pred_np).squeeze()
y_true_np = np.array(y_true_np).squeeze()
y_pred_prob_np = np.array(y_pred_prob_np).squeeze()

# Save the prediction probabilities
np.save("mc_output_" + str(num_mc) + "_probabilities" + ".npy", y_pred_prob_np)

# Calculate weighted F1 score
f1 = f1_score(y_true_np, y_pred_np, average='weighted')
print("Weighted F1 score:", f1)

# Calculate weighted recall
recall = recall_score(y_true_np, y_pred_np, average='weighted')
print("Weighted recall:", recall)

# Compute AUC for each class
unique_classes = np.unique(y_true_np)

if len(unique_classes) > 1:
    try:
        # Compute AUC score
        auc_score = roc_auc_score(y_true_np, y_pred_prob_np[:, 1], average='weighted')
        print("AUC:", auc_score)

        # --- Add ROC Curve Calculation and Plotting ---
        fpr, tpr, thresholds = roc_curve(y_true_np, y_pred_prob_np[:, 1])  # Compute ROC values
        roc_auc = auc(fpr, tpr)  # Compute the area under the curve (AUC)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig("ROC_Curve.pdf", dpi=300)  # Save the plot
        plt.show()
        print("ROC curve plotted and saved.")
    except ValueError as e:
        print(f"Error in computing AUC or plotting ROC curve: {e}")
else:
    print(f"AUC cannot be computed because y_true contains only one class: {unique_classes}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_np, y_pred_np)
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.savefig("Confusion Matrix.pdf", dpi=300)