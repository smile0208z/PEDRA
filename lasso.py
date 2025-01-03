import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

pca_path = r'D:\Survival_Analysis_Data\Processed_Data\e_den_pca.csv'
header_path = r'D:\Survival_Analysis_Data\combined_first_three_columns.csv'
output_path = r'D:\Survival_Analysis_Data\Processed_Data\e_den_lasso.csv'

df = pd.read_csv(pca_path)
X = df.drop(['patients', 'status', 'days'], axis=1)
y = df['status']

scaled_X = StandardScaler().fit_transform(X)
lasso = LassoCV(cv=10, random_state=0).fit(scaled_X, y)
selected_features = X.columns[lasso.coef_ != 0]
important_data = df[selected_features]

header = pd.read_csv(header_path)
combined = pd.concat([header, important_data], axis=1)
combined.to_csv(output_path, index=False)
print(combined)
