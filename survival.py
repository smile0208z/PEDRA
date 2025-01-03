import os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import brier_score_loss
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Survival_Analysis_Data\Processed_Data\e_den_lasso.csv')
y = df[['status', 'days']]
X = df.drop(['patients', 'days', 'status'], axis=1)
patients = df['patients']

X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(X, y, patients, test_size=0.3, random_state=42)

cph_uni, selected = CoxPHFitter(), []
for col in X_tr.columns:
    temp = pd.concat([X_tr[col], y_tr], axis=1)
    cph_uni.fit(temp, duration_col='days', event_col='status')
    if cph_uni.summary['p'][0] < 0.05: selected.append(col)

cph_multi = CoxPHFitter()
train_df = pd.concat([y_tr, X_tr[selected]], axis=1)
cph_multi.fit(train_df, duration_col='days', event_col='status')
final_cols = cph_multi.summary[cph_multi.summary['p'] < 0.05].index.tolist()

final_model = CoxPHFitter()
final_model.fit(pd.concat([y_tr, X_tr[final_cols]], axis=1), duration_col='days', event_col='status')
final_model.print_summary()

c_idx = final_model.concordance_index_
print(f'C-index: {c_idx:.4f}')

surv_pred = final_model.predict_survival_function(X).T
train_pred = final_model.predict_survival_function(X_tr).T
test_pred = final_model.predict_survival_function(X_te).T

def extract_prob(df_pred, patients, status, days, time=60.0):
    df = pd.DataFrame(df_pred[time], columns=[time])
    df['patients'], df['status'], df['days'] = patients.values, status.values, days.values
    return df

prob_60 = extract_prob(surv_pred, patients, y['status'], y['days'])
train_prob_60 = extract_prob(train_pred, p_tr, y_tr['status'], y_tr['days'])
test_prob_60 = extract_prob(test_pred, p_te, y_te['status'], y_te['days'])

prob_60.to_excel(r'C:\Users\z3322\Desktop\zhuomian\predicted_survival_PEDRA_60.xlsx', index=False)
train_prob_60.to_excel(r'C:\Users\z3322\Desktop\zhuomian\train_predicted_survival_PEDRA_60.xlsx', index=False)
test_prob_60.to_excel(r'C:\Users\z3322\Desktop\zhuomian\test_predicted_survival_PEDRA_60.xlsx', index=False)

b_score = {
    'data': brier_score_loss(prob_60['status'], 1 - prob_60[60.0]),
    'train': brier_score_loss(train_prob_60['status'], 1 - train_prob_60[60.0]),
    'test': brier_score_loss(test_prob_60['status'], 1 - test_prob_60[60.0])
}
mean_bs = np.mean([b_score['train'], b_score['test']])
print(f"Dataset BS: {b_score['data']:.4f}\nTrain BS: {b_score['train']:.4f}\nTest BS: {b_score['test']:.4f}\nMean BS: {mean_bs:.4f}")
