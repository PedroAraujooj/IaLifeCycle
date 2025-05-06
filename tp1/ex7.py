import seaborn as sns
from sklearn.model_selection import train_test_split

penguins = sns.load_dataset("penguins")

penguins = penguins.dropna()

penguins["is_adelie"] = (penguins["species"] == "Adelie").astype(int)

X = penguins.drop(columns="species")
y = penguins["is_adelie"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho da validação: {X_val.shape[0]} amostras")
