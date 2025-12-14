import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option("display.max_columns", None)
df = pd.read_csv("dataset/train.csv")

for idx, row in df.iterrows() :
    if row["label"] == "M":
        df.loc[idx,"label"] = 1
    else :
        df.loc[idx,"label"] = 0
print(df[["label"]])

from sklearn.preprocessing import StandardScaler, MinMaxScaler
standardization = StandardScaler()

df_data = standardization.fit_transform(df.drop(columns=["label", "id"]))
df = pd.DataFrame(df_data, columns=df.columns[2:])

corr_matrix = df.corr()
# Création de la heatmap avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)

# Affichage de la figure
plt.title('Matrice de Corrélation')
plt.show()