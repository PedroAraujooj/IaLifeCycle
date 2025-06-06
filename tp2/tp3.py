import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from tp2 import carrega_processa


def avalia_modelo(
        nome: str,
        estimator,
        param_grid: dict,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_folds: int = 5,
):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit="f1",
        n_jobs=-1,
        return_train_score=False,
    )
    gs.fit(X_train, y_train)

    cv_media = {
        m: gs.cv_results_[f"mean_test_{m}"][gs.best_index_]
        for m in scoring
    }

    best_est = gs.best_estimator_
    y_pred_holdout = best_est.predict(X_test)

    holdout = {
        "accuracy": accuracy_score(y_test, y_pred_holdout),
        "precision": precision_score(y_test, y_pred_holdout),
        "recall": recall_score(y_test, y_pred_holdout),
        "f1": f1_score(y_test, y_pred_holdout),
    }

    print(f"\n=== {nome} ===")
    print("Melhores hiperparâmetros:", gs.best_params_)
    print("\nMÉDIAS de CV (sem retraining):")
    for k, v in cv_media.items():
        print(f"  {k:<9}: {v:.4f}")

    print("\nHOLD-OUT (após retraining final):")
    for k, v in holdout.items():
        print(f"  {k:<9}: {v:.4f}")

    return best_est, cv_media, holdout, gs.best_params_


def main() -> None:
    X_train, X_test, y_train, y_test = carrega_processa()

    knn_grid = {"n_neighbors": list(range(1, 31))}
    knn_best, knn_cv, knn_hold, knn_params = avalia_modelo(
        "KNN",
        KNeighborsClassifier(),
        knn_grid,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    log_grid = {
        "C": np.logspace(-3, 3, 7),
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    }
    log_best, log_cv, log_hold, log_params = avalia_modelo(
        "Regressão Logística",
        LogisticRegression(max_iter=10_000),
        log_grid,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\n====== COMPARAÇÃO FINAL ======")
    chave = "f1"
    print(
        f"F1 (CV)  – KNN: {knn_cv[chave]:.4f} | LogReg: {log_cv[chave]:.4f}"
    )
    print(
        f"F1 (Hold) – KNN: {knn_hold[chave]:.4f} | LogReg: {log_hold[chave]:.4f}"
    )

    vencedor = (
        "Regressão Logística"
        if log_hold[chave] > knn_hold[chave]
        else "KNN"
    )
    print(f"\n>>> Modelo vencedor: {vencedor}\n")


if __name__ == "__main__":
    main()
