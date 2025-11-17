import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
df = pd.read_csv("train.csv")
# print(df.head())

# shape
shape = df.shape
print(f"shape : {shape} ")
print("\n")

# describe
description = df.describe()
# print(f"description : {description}") 
print("\n")

# proportion of M and B
proportion = df['label'].value_counts()
print(f"number of each label : {proportion}")
print("\n")

proportion = df['label'].value_counts(normalize=True)
print(f"proportion of each label : {proportion}")
print("\n")

#number of null values in the data set for each column
for column in df.columns : 
    number_of_null = df[column].isnull().sum()
    print(f" {column} : {number_of_null}")
print("\n")

# turn M to 1 and B to 0 so it can show using a graph
for idx, row in df.iterrows() :
    if row["label"] == "M":
        df.loc[idx,"turn"] = 1
    else :
        df.loc[idx,"turn"] = 0
print(df[["label","turn"]])

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


# # scatter plot using turn so we can see the range for each breast cancer or not
# for col in df.columns:
#     plt.figure(figsize=(6,4))
#     plt.scatter(
#         df[col], 
#         df["turn"], 
#     )
#     plt.title(f"Nuage de points : {col}")
#     plt.xlabel(f"Values of {col} ")
#     plt.ylabel("label : M = 1 and B = 0")
# plt.show()

