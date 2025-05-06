import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

penguins = sns.load_dataset("penguins")

penguins = penguins.dropna()

penguins["is_adelie"] = (penguins["species"] == "Adelie").astype(int)

#X = penguins.drop(columns="species")
numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = penguins[numeric_features]
y = penguins["is_adelie"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_val)

print("Acurácia:", accuracy_score(y_val, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred, target_names=["Não-Adelie", "Adelie"]))
