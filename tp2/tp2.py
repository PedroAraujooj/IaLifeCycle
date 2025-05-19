import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def carrega_processa(test_size: float = 0.2, random_state: int | None = 42):
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, y_train, y_test


def carrega_processa_com_ruido(
        test_size: float = 0.2,
        random_state: int | None = 42,
        noise_std_frac: float = 0.05,
        synth_factor: int = 1,
):
    raw = load_breast_cancer()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    rng = np.random.default_rng(seed=random_state)
    stds = X_train.std()

    synth_samples = []
    for _ in range(synth_factor):
        noise = rng.normal(
            loc=0.0,
            scale=stds.values * noise_std_frac,
            size=X_train.shape,
        )
        synth_samples.append(X_train.values + noise)

    if synth_samples:
        X_train_aug = pd.concat(
            [X_train]
            + [
                pd.DataFrame(s, columns=X_train.columns, index=X_train.index)
                for s in synth_samples
            ],
            axis=0,
            ignore_index=True,
        )
        y_train_aug = pd.concat(
            [y_train] * (1 + synth_factor), axis=0, ignore_index=True
        )
    else:
        X_train_aug, y_train_aug = X_train.copy(), y_train.copy()

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_aug),
        columns=X_train_aug.columns,
        index=X_train_aug.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train_aug, y_test


def carrega_processa_regressao(
        test_size: float = 0.2,
        random_state: int | None = 42,
):
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)

    y = df["mean area"]
    X = df.drop(columns=["mean area"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test


def treina_avalia_lr(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        verbose: bool = True,
):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    residuals = pd.Series(y_test - y_pred_test, index=y_test.index)

    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mean_residual": residuals.mean(),
        "std_residual": residuals.std(),
    }

    if verbose:
        print("\n▸ Regressão Linear – alvo: mean area")
        print(f"MAE   : {mae:.2f}")
        print(f"RMSE  : {rmse:.2f}")
        print(f"R²    : {r2:.4f}")
        print(f"Média resíduo : {metrics['mean_residual']:.2f}")
        print(f"DP resíduos   : {metrics['std_residual']:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y_pred_test, residuals, alpha=0.6)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set_xlabel("ŷ (predito)")
    ax[0].set_ylabel("resíduo")
    ax[0].set_title("Resíduos vs. valores preditos")

    ax[1].hist(residuals, bins=20, edgecolor="k")
    ax[1].set_xlabel("resíduo")
    ax[1].set_ylabel("freq.")
    ax[1].set_title("Distribuição dos resíduos")
    plt.tight_layout()
    plt.savefig("regressaoLinear.png")

    return model, metrics, residuals


def treina_avalia_knn(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        k: int = 5,
        verbose: bool = True,
):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics = {
        "k": k,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "classification_report": classification_report(
            y_test, y_pred_test, target_names=["malignant", "benign"]
        ),
    }

    if verbose:
        print(f"\n▸ Resultados para k={k}")
        print("Acurácia treino:", metrics["train_accuracy"])
        print("Acurácia teste :", metrics["test_accuracy"])
        print("Matriz de confusão: 1ª linha Malignos  e 2ª Benignos  \n", metrics["confusion_matrix"])
        print("Relatório completo:\n", metrics["classification_report"])

    return model, metrics


def analisa_var_k(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        k_min: int = 1,
        k_max: int = 20,
        plot: bool = True,
        com_ruido: bool = False,
):
    ks, train_acc, test_acc = [], [], []

    for k in range(k_min, k_max + 1):
        _, m = treina_avalia_knn(
            X_train, X_test, y_train, y_test, k=k, verbose=False
        )
        ks.append(k)
        train_acc.append(m["train_accuracy"])
        test_acc.append(m["test_accuracy"])

    df = pd.DataFrame(
        {"k": ks, "train_accuracy": train_acc, "test_accuracy": test_acc}
    )

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(df["k"], df["train_accuracy"], "o--", label="Treino")
        plt.plot(df["k"], df["test_accuracy"], "o-", label="Teste")
        plt.xlabel("Número de vizinhos (k)")
        plt.ylabel("Acurácia")
        if com_ruido:
            plt.title("KNN – Acurácia vs. k - COM RUÍDO")
        else:
            plt.title("KNN – Acurácia vs. k - SEM RUÍDO")
        plt.xticks(df["k"])
        plt.grid(True, ls=":")
        plt.legend()
        plt.tight_layout()
        if com_ruido:
            plt.savefig("COM ruido.png")
        else:
            plt.savefig("sem ruido.png")

    return df


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = carrega_processa()
    X_train_scaled_ruido, X_test_scaled_ruido, y_train_ruido, y_test_ruido = carrega_processa_com_ruido()

    print("=========================Resultado sem ruido==========================")
    df_k = analisa_var_k(X_train_scaled, X_test_scaled, y_train, y_test, k_min=1, k_max=20, com_ruido=False)

    best_row = df_k.loc[df_k["test_accuracy"].idxmax()]
    best_k = int(best_row["k"])
    print(f"\nMelhor k = {best_k} | acurácia teste = {best_row['test_accuracy']:.4f}")

    _, _ = treina_avalia_knn(X_train_scaled, X_test_scaled, y_train, y_test, k=best_k)

    print("=========================Resultado COM ruido==========================")
    df_k_ruido = analisa_var_k(X_train_scaled_ruido, X_test_scaled_ruido, y_train_ruido, y_test_ruido, k_min=1,
                               k_max=20, com_ruido=True)

    best_row_ruido = df_k_ruido.loc[df_k_ruido["test_accuracy"].idxmax()]
    best_k_ruido = int(best_row_ruido["k"])
    print(f"\nMelhor k = {best_k_ruido} | acurácia teste = {best_row_ruido['test_accuracy']:.4f}")

    _, _ = treina_avalia_knn(X_train_scaled_ruido, X_test_scaled_ruido, y_train_ruido, y_test_ruido, k=best_k_ruido)

    print("=========================REGRESSÃO LINEAR==========================")
    Xtr_r, Xte_r, ytr_r, yte_r = carrega_processa_regressao()
    _, reg_metrics, _ = treina_avalia_lr(
        Xtr_r, Xte_r, ytr_r, yte_r, verbose=True
    )
