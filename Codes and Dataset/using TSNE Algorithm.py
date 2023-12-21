import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'CO2 Emissions.csv'
file_path = os.path.join(script_dir, file_name)
df = pd.read_csv(file_path)

categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

label_encoder = LabelEncoder()

for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])
tsne = TSNE(n_components=2)
reduced_data = tsne.fit_transform(df)


plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('t-SNE Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

