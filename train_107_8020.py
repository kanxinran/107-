"""
107-dim BiLSTM (99+C+E) — URFD+Le2i Fall Detection.
Train=80%, Test=20%, no separate validation.

Usage:
    python train_107_8020.py

Requires pre-extracted .npy data files in ../data/:
    urfd_X_99.npy, urfd_y.npy, le2i_X_99.npy, le2i_y.npy
"""
import os, sys, csv, time, json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pose_extractor import PoseExtractor
from feature_engine import compute_features
from anatomical_features import compute_anatomical_features, FEATURE_SLICES
from visualizer import draw_pose_skeleton, draw_status_overlay

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(THIS_DIR), "data")  # ../data/
CSV_DIR = os.path.join(THIS_DIR, "csv")
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW, INPUT_DIM, EPOCHS = 30, 107, 100
INIT_LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | Input: {INPUT_DIM}d (99 + C4 + E4) | Epochs: {EPOCHS} | LR: {INIT_LR}")

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading cached 99-dim data...")
urfd_X = np.load(os.path.join(DATA_DIR, "urfd_X_99.npy"))
urfd_y = np.load(os.path.join(DATA_DIR, "urfd_y.npy"))
le2i_X = np.load(os.path.join(DATA_DIR, "le2i_X_99.npy"))
le2i_y = np.load(os.path.join(DATA_DIR, "le2i_y.npy"))

X99 = np.concatenate([urfd_X, le2i_X], axis=0).astype(np.float32)
y = np.concatenate([urfd_y, le2i_y], axis=0).astype(np.int64)
print(f"Total: {X99.shape[0]} windows | Fall: {(y==1).sum()} | ADL: {(y==0).sum()}")

# Remove all-zero
valid = ~np.all(X99.reshape(len(X99), -1) == 0, axis=1)
X99, y = X99[valid], y[valid]
print(f"Valid: {len(X99)} (removed {(~valid).sum()})")

# ============================================================
# COMPUTE C+E FEATURES
# ============================================================
print("\nComputing C (velocity) and E (stability) features...")
t0 = time.time()
N = len(X99)
C_feats = np.zeros((N, WINDOW, 4), dtype=np.float32)
E_feats = np.zeros((N, WINDOW, 4), dtype=np.float32)

for i in range(N):
    kps_win = X99[i].reshape(WINDOW, 33, 3)
    for t in range(WINDOW):
        kps_t = kps_win[t]
        prev = kps_win[t-1] if t > 0 else None
        feats_21 = compute_features(kps_t, prev)
        C_feats[i, t] = feats_21[9:13]
        E_feats[i, t] = feats_21[15:19]
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{N} ({time.time()-t0:.1f}s)")

# Build 107-dim: 99 raw + C(4) + E(4)
X107 = np.concatenate([X99, C_feats.reshape(N, WINDOW, 4), E_feats.reshape(N, WINDOW, 4)], axis=2)
print(f"107-dim built in {time.time()-t0:.1f}s | shape: {X107.shape}")

# Shuffle
shuffle_idx = np.random.RandomState(42).permutation(len(X107))
X107, y = X107[shuffle_idx], y[shuffle_idx]

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X107, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTrain: {X_train.shape[0]} | Fall: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} | Fall: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# Normalize (fit on train only)
mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1)) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
np.save(os.path.join(THIS_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(THIS_DIR, "norm_std.npy"), std)

# ============================================================
# MODEL
# ============================================================
class FallLSTM(nn.Module):
    def __init__(self, input_dim=107, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.attention = nn.Linear(hidden_dim*2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)
        return self.classifier(context)

# ============================================================
# DATALOADERS
# ============================================================
class FallDataset(Dataset):
    def __init__(self, Xd, yd):
        self.X = torch.FloatTensor(Xd); self.y = torch.LongTensor(yd)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class_counts = np.bincount(y_train, minlength=2)
weights = 1.0 / class_counts
sampler = WeightedRandomSampler(weights[y_train], len(y_train), replacement=True)
train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=64, sampler=sampler)
test_loader = DataLoader(FallDataset(X_test, y_test), batch_size=64, shuffle=False)

print(f"\nTrain batches: {len(train_loader)} | Test batches: {len(test_loader)}")

# ============================================================
# TRAIN
# ============================================================
model = FallLSTM(input_dim=INPUT_DIM).to(DEVICE)
fall_weight = class_counts[0] / max(class_counts[1], 1)
print(f"Fall class weight: {fall_weight:.2f}x | Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, fall_weight]).to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5)

def evaluate(loader):
    model.eval()
    preds, trues, probs_list = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(DEVICE))
            total_loss += criterion(logits, y_b.to(DEVICE)).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y_b.numpy())
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    cm = confusion_matrix(trues, preds)
    TN, FP, FN, TP = cm.ravel()
    return {
        'loss': total_loss / len(loader),
        'accuracy': float(accuracy_score(trues, preds)),
        'f1_weighted': float(f1_score(trues, preds, average='weighted', zero_division=0)),
        'fall_recall': float(recall_score(trues, preds, pos_label=1, zero_division=0)),
        'fall_precision': float(precision_score(trues, preds, pos_label=1, zero_division=0)),
        'fall_f1': float(f1_score(trues, preds, pos_label=1, zero_division=0)),
        'auc': float(roc_auc_score(trues, probs[:, 1])),
        'fpr': float(FP/(FP+TN+1e-8)),
        'specificity': float(TN/(TN+FP+1e-8)),
        'confusion': f"TN={TN} FP={FP} FN={FN} TP={TP}",
    }

best_test_f1 = 0.0
best_state = None
history = []

print("\nTraining...")
t_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_preds, train_true = [], []
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        train_preds.extend(logits.argmax(1).cpu().numpy())
        train_true.extend(y_b.cpu().numpy())

    train_acc = np.mean(np.array(train_preds) == np.array(train_true))
    avg_train_loss = train_loss / len(train_loader)

    test_metrics = evaluate(test_loader)
    test_f1 = test_metrics['fall_f1']
    scheduler.step(test_f1)

    lr = optimizer.param_groups[0]['lr']
    history.append({
        'epoch': epoch,
        'train_loss': round(avg_train_loss, 6),
        'train_acc': round(train_acc, 6),
        'test_loss': round(test_metrics['loss'], 6),
        'test_acc': round(test_metrics['accuracy'], 6),
        'test_f1w': round(test_metrics['f1_weighted'], 6),
        'test_fallrec': round(test_metrics['fall_recall'], 6),
        'test_fallprec': round(test_metrics['fall_precision'], 6),
        'test_fpr': round(test_metrics['fpr'], 6),
        'test_auc': round(test_metrics['auc'], 6),
        'test_fallf1': round(test_metrics['fall_f1'], 6),
        'lr': round(lr, 10),
    })

    if test_f1 > best_test_f1:
        best_test_f1 = test_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch

    mark = "  <- best" if test_f1 == best_test_f1 else ""
    print(f"Epoch {epoch:3d} | TrainLoss {avg_train_loss:.4f} | TrainAcc {train_acc:.3f} | "
          f"TestAcc {test_metrics['accuracy']:.4f} | F1w {test_metrics['f1_weighted']:.4f} | "
          f"FallRec {test_metrics['fall_recall']:.4f} | FallPrec {test_metrics['fall_precision']:.4f} | "
          f"FPR {test_metrics['fpr']:.4f}{mark}")

print(f"\nTraining done in {time.time()-t_start:.1f}s | Best Test F1: {best_test_f1:.4f} at epoch {best_epoch}")

# ============================================================
# SAVE
# ============================================================
csv_path = os.path.join(CSV_DIR, "training_history.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=history[0].keys())
    w.writeheader()
    w.writerows(history)

model.load_state_dict(best_state)
model_path = os.path.join(THIS_DIR, "fall_lstm_107_best.pt")
torch.save(best_state, model_path)

final = evaluate(test_loader)
results_path = os.path.join(THIS_DIR, "results_107.json")
with open(results_path, 'w') as f:
    json.dump({
        "config": f"107-dim (99+C+E) URFD+Le2i 80/20 LR={INIT_LR}",
        "input_dim": 107, "epochs": EPOCHS, "best_epoch": best_epoch,
        "init_lr": INIT_LR,
        **{k: v for k, v in final.items() if k != 'confusion'},
        "confusion": final['confusion'],
        "params": sum(p.numel() for p in model.parameters()),
    }, f, indent=2)

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS (107-dim: 99+C+E)")
print(f"{'='*60}")
print(f"Accuracy:     {final['accuracy']:.4f}")
print(f"F1 Weighted:  {final['f1_weighted']:.4f}")
print(f"Fall Recall:  {final['fall_recall']:.4f}")
print(f"Fall Precision:{final['fall_precision']:.4f}")
print(f"Fall F1:      {final['fall_f1']:.4f}")
print(f"AUC:          {final['auc']:.4f}")
print(f"FPR:          {final['fpr']:.4f}")
print(f"Specificity:  {final['specificity']:.4f}")
print(f"Confusion:    {final['confusion']}")

# ============================================================
# PLOT
# ============================================================
epochs_list = [h['epoch'] for h in history]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes[0,0].plot(epochs_list, [h['train_loss'] for h in history], 'b-', label='Train')
axes[0,0].plot(epochs_list, [h['test_loss'] for h in history], 'r-', label='Test')
axes[0,0].set_title("Loss"); axes[0,0].legend()

axes[0,1].plot(epochs_list, [h['train_acc'] for h in history], 'b-', label='Train')
axes[0,1].plot(epochs_list, [h['test_acc'] for h in history], 'r-', label='Test')
axes[0,1].set_title("Accuracy"); axes[0,1].legend()

axes[0,2].plot(epochs_list, [h['test_f1w'] for h in history], 'g-')
axes[0,2].set_title("Test F1 Weighted")

axes[0,3].plot(epochs_list, [h['lr'] for h in history], 'm-')
axes[0,3].set_title("Learning Rate")

axes[1,0].plot(epochs_list, [h['test_fallrec'] for h in history], 'r-', label='Recall')
axes[1,0].plot(epochs_list, [h['test_fallprec'] for h in history], 'b-', label='Precision')
axes[1,0].set_title("Fall Recall & Precision"); axes[1,0].legend()

axes[1,1].plot(epochs_list, [h['test_fpr'] for h in history], 'r-')
axes[1,1].set_title("FPR (False Positive Rate)")

axes[1,2].plot(epochs_list, [h['test_auc'] for h in history], 'purple')
axes[1,2].set_title("AUC")

axes[1,3].plot(epochs_list, [h['test_fallf1'] for h in history], 'r-')
axes[1,3].set_title("Test Fall F1")

plt.suptitle(f"107-dim (99+C+E) BiLSTM | URFD+Le2i 80/20 | LR={INIT_LR} | Best FallF1={best_test_f1:.4f} (Epoch {best_epoch})")
plt.tight_layout()
fig.savefig(os.path.join(THIS_DIR, "training_curves.png"), dpi=150)
plt.close()
print(f"\nPlot saved.")

print(f"\nAll done. Files in: {THIS_DIR}")
