"""
main.py : orchestre le pipeline ML (Adult Census Income) via arguments CLI.

Exemples :
    python main.py --action train
    python main.py --action evaluate
    python main.py --action predict
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


# --------------------- Fonctions d'étapes (réutilisées par Prefect) ---------------------
def step_prepare(data_path):
    """Charge et prétraite les données."""
    return prepare_data(data_path)


def step_train(X_train, y_train):
    """Entraîne le modèle."""
    return train_model(X_train, y_train)


def step_save(model, preprocessor, model_path, preprocessor_path):
    """Sauvegarde le modèle et le préprocesseur."""
    save_model(model, model_path)
    save_model(preprocessor, preprocessor_path)


def step_load(model_path):
    """Charge le modèle sauvegardé."""
    return load_model(model_path)


def step_evaluate(model, X_test, y_test):
    """Évalue le modèle, affiche et retourne les métriques."""
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}"
        )
    return metrics


# ------------------------------------- CLI -------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning - Adult Census Income"
    )
    parser.add_argument(
        "--action",
        choices=["train", "evaluate", "predict"],
        required=True,
        help="train : entraîne, évalue et sauvegarde | "
        "evaluate : évalue le modèle sauvegardé | "
        "predict : prédit sur quelques exemples du jeu de test",
    )
    parser.add_argument(
        "--data", default="adult.data", help="Chemin vers le fichier de données"
    )
    parser.add_argument(
        "--model",
        default="random_forest_model.joblib",
        help="Chemin du modèle sauvegardé",
    )
    parser.add_argument(
        "--preprocessor",
        default="preprocessor.joblib",
        help="Chemin du préprocesseur sauvegardé",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Nombre d'exemples pour --action predict"
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, preprocessor = step_prepare(args.data)

    if args.action == "train":
        model = step_train(X_train, y_train)
        print("\n===== Évaluation =====")
        step_evaluate(model, X_test, y_test)
        step_save(model, preprocessor, args.model, args.preprocessor)
        print(f"\nModèle sauvegardé : {args.model}")
        print("Pipeline terminé avec succès !")

    elif args.action == "evaluate":
        model = step_load(args.model)
        step_evaluate(model, X_test, y_test)

    elif args.action == "predict":
        model = step_load(args.model)
        sample = X_test.head(args.n)
        preds = model.predict(sample)
        probas = model.predict_proba(sample)[:, 1]
        labels = {0: "<=50K", 1: ">50K"}
        for i, (p, pr, real) in enumerate(zip(preds, probas, y_test.head(args.n))):
            print(
                f"Exemple {i + 1}: prédit {labels[p]} (proba >50K = {pr:.2f}) | réel {labels[real]}"
            )


if __name__ == "__main__":
    main()
