from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os
import plotly.express as px

"""
label:
0 normal
1 fraud
0.5 public dataset (or private dataset, after the release of the labels of the public dataset)
"""

TRAIN_DATA = "./train_clean.csv"
TEST_DATA = "./public_clean.csv"
OUT_CSV = "./embedding.csv"
EMBED_DIM = 3

train_df = pd.read_csv(TRAIN_DATA, engine='pyarrow')
train_df = train_df.sort_values(by=['label'], ascending=False)[20000:40000]
print("train:", TRAIN_DATA)
test_df = pd.read_csv(TEST_DATA, engine='pyarrow')[:1000]
print("test: ", TEST_DATA)

test_label = pd.DataFrame(np.full(len(test_df), 0.5), columns=['label'])
test_df = pd.concat((test_df, test_label), axis=1)

feature_extraction = pd.concat((train_df, test_df), axis=0)
print(feature_extraction)

# Reset index of feature_extraction
feature_extraction.reset_index(drop=True, inplace=True)

# Feature Extraction: TSNE
embedding = TSNE(n_components=EMBED_DIM, 
                 verbose=True, 
                 n_iter=250,
                 init="random",
                 learning_rate="auto",
                 n_jobs=os.cpu_count()-5).fit_transform(feature_extraction.drop(columns=['txkey', 'label']))

# 2D/3D embedding space plot
fig = px.scatter_3d(
    embedding, x=0, y=1, z=2,
    color=feature_extraction['label'], labels={'color': 'label'}
)
fig.show()

"""
# Create DataFrame from embedding
temp_df = pd.DataFrame({'embedding1': embedding[:, 0:1].flatten(),
                        'embedding2': embedding[:, 1:2].flatten(),
                        'embedding3': embedding[:, 2:3].flatten()})
temp_df.reset_index(drop=True, inplace=True)  # Reset index to ensure it's unique

# Concatenate DataFrames
new = pd.concat([feature_extraction[['txkey', 'label']], temp_df], axis=1)
new.to_csv(OUT_CSV, index=False)

"""
