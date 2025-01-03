import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_evaluate_model(m, tl, vl, te_l, c, o, ne, d, p=10, od='', n=''):
    tr_l, vl_l = [], []
    tr_a, vl_a = [], []
    tr_auc, vl_auc = [], []
    best_vl = float('inf')
    e_stop = 0
    metrics = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train Acc', 'Train AUC', 'Val Loss', 'Val Acc', 'Val AUC'])

    def eval_m(dl):
        m.eval()
        loss_tot, labels, preds = 0.0, [], []
        with torch.no_grad():
            for data, label in dl:
                data, label = data.to(d), label.to(d)
                _, out = m(data.float())
                loss_tot += c(out, label.float()).item()
                labels += label.cpu().numpy().tolist()
                preds += out.sigmoid().cpu().numpy().tolist()
        return loss_tot, accuracy_score(labels, np.round(preds)), roc_auc_score(labels, preds)

    for epoch in range(ne):
        m.train()
        loss_tot, labels, preds = 0.0, [], []
        pb = tqdm(enumerate(tl), total=len(tl), desc=f"Epoch {epoch+1}/{ne}")
        for i, (data, label) in pb:
            data, label = data.to(d).float(), label.to(d)
            _, out = m(data)
            loss = c(out, label.float())
            loss.backward()
            loss_tot += loss.item()
            labels += label.cpu().numpy().tolist()
            preds += out.sigmoid().cpu().detach().numpy().tolist()
            o.step()
            o.zero_grad()
        tr_l.append(loss_tot)
        tr_a.append(accuracy_score(labels, np.round(preds)))
        tr_auc.append(roc_auc_score(labels, preds))
        vl_l_val, vl_acc, vl_auc_val = eval_m(vl)
        vl_l.append(vl_l_val)
        vl_a.append(vl_acc)
        vl_auc.append(vl_auc_val)
        print(f"Epoch {epoch+1}: Train Loss {loss_tot:.4f}, Acc {tr_a[-1]:.4f}, AUC {tr_auc[-1]:.4f}")
        print(f"Epoch {epoch+1}: Val Loss {vl_l_val:.4f}, Acc {vl_acc:.4f}, AUC {vl_auc_val:.4f}")
        metrics.loc[epoch] = [epoch+1, loss_tot, tr_a[-1], tr_auc[-1], vl_l_val, vl_acc, vl_auc_val]
        if vl_l_val < best_vl:
            best_vl = vl_l_val
            e_stop = 0
        else:
            e_stop += 1
            if e_stop >= p:
                print(f"Early stopping at epoch {epoch+1}")
                break

    te_l_val, te_acc, te_auc_val = eval_m(te_l)
    print(f"Test: Loss {te_l_val:.4f}, Acc {te_acc:.4f}, AUC {te_auc_val:.4f}")
    metrics['Test Loss'] = te_l_val
    metrics['Test Acc'] = te_acc
    metrics['Test AUC'] = te_auc_val
    metrics.to_csv(f"{od}/metrics_{n}.csv", index=False)

    def p_s(tm, vm, mn, od, n):
        plt.figure(figsize=(10,5))
        plt.plot(tm, label=f'train {mn}')
        plt.plot(vm, label=f'val {mn}')
        plt.title(f'{mn} curve')
        plt.xlabel('epoch')
        plt.ylabel(mn)
        plt.legend()
        plt.savefig(f"{od}/{mn}_{n}.png")
    p_s(tr_l, vl_l, 'loss', od, n)
    p_s(tr_a, vl_a, 'accuracy', od, n)
    p_s(tr_auc, vl_auc, 'auc', od, n)

    return metrics