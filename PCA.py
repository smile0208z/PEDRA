import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and preprocess data
data_path = r'D:\Survival_Analysis_Data\Processed_Data\e_den_sum.csv'
df = pd.read_csv(data_path).drop(['patients', 'status', 'days'], axis=1)

# Define transformation pipeline
transform = Pipeline([
    ('scale', StandardScaler()),
    ('reduce', PCA(n_components=300))
])

# Apply transformations
transformed = transform.fit_transform(df)
transformed_df = pd.DataFrame(transformed)

# Integrate with header
header = pd.read_csv(r'D:\Survival_Analysis_Data\combined_first_three_columns.csv')
final_df = pd.concat([header, transformed_df], axis=1)

# Export result
final_df.to_csv(r'D:\Survival_Analysis_Data\Processed_Data\e_den_pca.csv', index=False)
print(final_df)
