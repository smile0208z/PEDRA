import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    confusion_matrix, auc, roc_curve, mean_squared_error,
    mean_absolute_error, r2_score, brier_score_loss, precision_recall_curve
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn.functional as F


class DeepSurv(nn.Module):
    def __init__(self, input_dim, dropout_rates=None, hidden_dims=None):
        super(DeepSurv, self).__init__()
        hidden_dims = hidden_dims or [256, 512, 256, 128]
        dropout_rates = dropout_rates or [0.4, 0.4, 0.3, 0.3]
        layers = []
        prev_dim = input_dim
        for hd, dr in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.LeakyReLU(0.1),
                nn.Dropout(dr)
            ])
            prev_dim = hd
        self.feature_extractor = nn.Sequential(*layers)
        self.residual_layers = nn.ModuleList([nn.Linear(hd, hd) for hd in hidden_dims])
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1]),
            nn.Sigmoid()
        )
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        for layer in self.residual_layers:
            residual = layer(out)
            out = F.relu(out + residual)
        attn = self.attention(out)
        out = out * attn
        return self.final_layers(out)


# Load the dataset
data = pd.read_csv(r'D:\生存分析数据\处理数据\32e_lasso.csv')

# Prepare features and labels
X = data.drop(columns=['patients', 'days', 'status'])  # Feature vectors
y = data['status']  # Binary classification labels

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X.values)
y_tensor = torch.FloatTensor(y.values).view(-1, 1)

# Initialize lists to store evaluation metrics
epochs = 90
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
accuracies = []
sensitivities = []
specificities = []
mcc_scores = []
brier_scores = []

save_dir = r'C:\Users\z3322\Desktop\zhuomian\生存预测'

fig_roc, ax_roc = plt.subplots()

skf = StratifiedKFold(n_splits=10, shuffle=True)

sum_confusion_matrix = np.zeros((2, 2))

# Iterate over each fold
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    # Split data into training and testing for the current fold
    X_train_fold, X_test_fold = X_tensor[train_index], X_tensor[test_index]
    y_train_fold, y_test_fold = y_tensor[train_index], y_tensor[test_index]

    model = DeepSurv(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_fold)
        loss = criterion(outputs, y_train_fold)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_probs = torch.sigmoid(model(X_test_fold)).numpy().flatten()
        y_pred_labels = (y_pred_probs > 0.5).astype(int)

    current_confusion_matrix = confusion_matrix(y_test_fold.numpy(), y_pred_labels)
    sum_confusion_matrix += current_confusion_matrix

    bs = brier_score_loss(y_test_fold.numpy(), y_pred_probs)
    brier_scores.append(round(bs, 4))

    cm = confusion_matrix(y_test_fold.numpy(), y_pred_labels)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])

    # Add numerical labels to each cell in the confusion matrix
    for (k, j), val in np.ndenumerate(cm):
        ax.text(j, k, val, ha='center', va='center',
                color='white' if cm[k, j] > cm.max() / 2 else 'black')

    plt.title(f'Confusion Matrix for Fold {i + 1}')
    plt.savefig(os.path.join(save_dir, f'fold_{i + 1}.png'))  # Save as PNG
    plt.close(fig)  # Close the figure to free memory

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_fold.numpy(), y_pred_probs)
    tprs.append(np.interp(mean_fpr, fpr, tpr))  # Interpolate TPR
    tprs[-1][0] = 0.0
    tprs[-1][-1] = 1.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    accuracies.append(round(accuracy_score(y_test_fold.numpy(), y_pred_labels), 4))
    tn, fp, fn, tp = cm.ravel()
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    specificities.append(specificity)

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator != 0 else 0
    mcc_scores.append(round(mcc, 4))

    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    sensitivities.append(sensitivity)

    ax_roc.plot(fpr, tpr, lw=1, alpha=0.3)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax_roc.plot(
    mean_fpr, mean_tpr,
    color='b',
    label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})',
    lw=2,
    alpha=0.8
)
ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)  # Diagonal line

ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set(xlim=[0, 1], ylim=[0, 1], title="Receiver Operating Characteristic")
ax_roc.legend(loc="lower right")
plt.show()

accuracy_str = ", ".join([f"{acc:.4f}" for acc in accuracies])
sensitivity_str = ", ".join([f"{sens:.4f}" for sens in sensitivities])
specificity_str = ", ".join([f"{spec:.4f}" for spec in specificities])
mcc_str = ", ".join([f"{mcc:.4f}" for mcc in mcc_scores])
auc_str = ", ".join([f"{auc_score:.4f}" for auc_score in aucs])
brier_score_str = ", ".join([f"{bs:.4f}" for bs in brier_scores])

# Print evaluation metrics
print(f'Accuracy: {accuracy_str}')
print(f'Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')

print(f'Sensitivity: {sensitivity_str}')
print(f'Mean Sensitivity: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}')

print(f'Specificity: {specificity_str}')
print(f'Mean Specificity: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}')

print(f'MCC Scores: {mcc_str}')
print(f'Mean MCC: {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}')

print(f'AUC Scores: {auc_str}')
print(f'Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')

print(f'Brier Scores: {brier_score_str}')
print(f'Mean Brier Score: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}')

# Compute the average confusion matrix
conf_matrix = sum_confusion_matrix

fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['0', '1'])
ax.set_yticklabels(['0', '1'])

# Add numerical labels to each cell in the confusion matrix
for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, int(val), ha='center', va='center', color='black')

plt.title('Confusion Matrix')
plt.show()

# Create a DataFrame with the collected metrics
results = pd.DataFrame({
    'Accuracy': accuracies,
    'Specificity': specificities,
    'MCC': mcc_scores
})

results.to_csv(r'C:\Users\z3322\Desktop\zhuomian\生存预测\ev2.csv', index=False)
