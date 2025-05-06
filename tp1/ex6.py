import seaborn as sns

penguins = sns.load_dataset("penguins")

features = penguins.drop(columns=["species"])
target = penguins["species"]

print("\nFeatures:")
print(features.columns.tolist())

print("\nTarget:")
print(target.name)
