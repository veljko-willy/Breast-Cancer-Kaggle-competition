import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
df = pd.read_csv("dataset/train.csv")
# print(df.head())

# shape
# shape = df.shape
# print(f"shape : {shape} ")
# print("\n")

# # describe
# description = df.describe()
# # print(f"description : {description}") 
# print("\n")

# # proportion of M and B
# proportion = df['label'].value_counts()
# print(f"number of each label : {proportion}")
# print("\n")

# proportion = df['label'].value_counts(normalize=True)
# print(f"proportion of each label : {proportion}")
# print("\n")

# #number of null values in the data set for each column
# for column in df.columns : 
#     number_of_null = df[column].isnull().sum()
#     print(f" {column} : {number_of_null}")
# print("\n")

# turn M to 1 and B to 0 so it can show using a graph
for idx, row in df.iterrows() :
    if row["label"] == "M":
        df.loc[idx,"label"] = 1
    else :
        df.loc[idx,"label"] = 0
print(df[["label"]])

# # scatter plot using colour differenciation
# for col in df.columns:
#     plt.figure(figsize=(6,4))
#     plt.scatter(
#         df.index, 
#         df[col], 
#         c=df['label'].map({'B': 'blue', 'M': 'red'}), 
#     )
#     plt.title(f"Nuage de points : {col}")
#     plt.xlabel("Index (Begnin = Blue and Malignent = Red)")
#     plt.ylabel(col)
# plt.show()


from sklearn.preprocessing import StandardScaler, MinMaxScaler
standardization = StandardScaler()
min_max_scaler = MinMaxScaler()

corr_matrix = df.corr()
# print(corr_matrix)
# plt.figure(figsize=(33, 33))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix before Scaling")
# plt.show()

standardized_data = standardization.fit_transform(df.drop(columns=["label", "id"]))
min_max_scaled_data = min_max_scaler.fit_transform(df.drop(columns=["label", "id"]))
standardized_df = pd.DataFrame(standardized_data, columns=df.columns[2:])
min_max_scaled_df = pd.DataFrame(min_max_scaled_data, columns=df.columns[2:])

corr_matrix=standardized_df.corr()
# plt.figure(figsize=(33, 33))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix after Standardization")
# plt.show()

# take all the duos of parameters with a correlation higher than 0.9
high_corr_pairs = {}
checked_pairs = set()  # Set pour garder une trace des paires déjà vérifiées

for parameter1 in standardized_df.columns:
    for parameter2 in standardized_df.columns:
        if parameter1 != parameter2 and (parameter2, parameter1) not in checked_pairs:
            if corr_matrix.loc[parameter1, parameter2] > 0.9: 
                high_corr_pairs[parameter1] = parameter2
            # Ajouter la paire dans les deux directions pour éviter les duplications
            checked_pairs.add((parameter1, parameter2))

print(high_corr_pairs)

## Checker les distributions après différente normalisation
#check the distribution of each parameters
# for col in standardized_df.columns:
#     # Affichage de l'histogramme pour chaque colonne
#     plt.hist(standardized_df[col], bins=30, color='skyblue', edgecolor='black')  # dropna() pour ignorer les valeurs manquantes
#     plt.title(f"Distribution de la colonne {col} standardisée")
#     plt.xlabel(col)
#     plt.ylabel('Fréquence')
#     plt.show()

# for col in min_max_scaled_df.columns:
#     #Affichage de l'histogramme pour chaque colonne du df normalisé selon min max
#     plt.hist(min_max_scaled_df[col], bins = 30, color='red',edgecolor='black')
#     plt.title(f"Distribution de la colonne {col} min max normalisée")
#     plt.xlabel(col)
#     plt.ylabel('Fréquence')
#     plt.show()

## Mise en place et comparaisons de modèles avec des données standardisées
from sklearn import linear_model
model = linear_model.LogisticRegression()

print(standardized_df.head())

x_train = standardized_df.drop("label",axis=1)
y_train = standardized_df["label"]

df_test = pd.read_csv("dataset/train.csv")
df_sub = pd.read_csv("dataset/sample_submission.csv")

for idx, row in df_sub.iterrows() :
    if row["label"] == "M":
        df_sub.loc[idx,"label"] = 1
    else :
        df_sub.loc[idx,"label"] = 0

model.fit(x_train,y_train)

x_test = df_test.drop("id",axis=1)
y_test = df_test["label"]

y_predict=model.predict(x_test)
