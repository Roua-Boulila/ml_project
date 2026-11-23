"""
model_pipeline.py
-----------------
Pipeline ML modulaire pour le dataset Adult Census Income.
Prédiction : income <=50K (0) / >50K (1).

Fonctions :
    prepare_data()   -> charge et prétraite les données
    train_model()    -> entraîne le Random Forest
    evaluate_model() -> calcule les métriques
    save_model()     -> sauvegarde avec joblib
    load_model()     -> recharge un objet sauvegardé
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def prepare_data(path, test_size=0.20, random_state=42):
    """Charge et prétraite le dataset Adult.

    Étapes (identiques au notebook, section 4) :
      1. '?' -> NaN puis NaN -> 'Unknown'
      2. suppression des doublons
      3. suppression de 'education' et 'fnlwgt'
      4. regroupement de native-country (United-States / Other)
         et des modalités rares de workclass
      5. log1p sur capital-gain et capital-loss
      6. split train/test stratifié
      7. StandardScaler + OneHotEncoder (appris sur le train uniquement)

    Args:
        path (str): chemin vers le fichier adult.data.
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour la reproductibilité.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    df = pd.read_csv(path, names=COLUMNS, skipinitialspace=True)

    # 1) Valeurs manquantes
    df = df.replace("?", np.nan)
    for col in ["workclass", "occupation", "native-country"]:
        df[col] = df[col].fillna("Unknown")

    # 2) Doublons
    df = df.drop_duplicates()

    # 3) Variables redondantes / inutiles
    df = df.drop(columns=["education", "fnlwgt"])

    # 4) Regroupement des modalités
    df["native-country"] = np.where(
        df["native-country"].eq("United-States"), "United-States", "Other"
    )
    counts = df["workclass"].value_counts()
    rare = counts[counts < 100].index
    df["workclass"] = df["workclass"].where(~df["workclass"].isin(rare), "Other")

    # 5) Asymétrie
    df["capital-gain"] = np.log1p(df["capital-gain"])
    df["capital-loss"] = np.log1p(df["capital-loss"])

    # 6) Cible + split stratifié
    y = (df["income"] == ">50K").astype(int)
    X = df.drop(columns=["income"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7) Encodage / scaling (fit sur le train uniquement : pas de fuite)
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_p = pd.DataFrame(X_train_p, columns=feature_names, index=X_train.index)
    X_test_p = pd.DataFrame(X_test_p, columns=feature_names, index=X_test.index)

    return X_train_p, X_test_p, y_train, y_test, preprocessor


def train_model(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=16,
    min_samples_leaf=3,
    random_state=42,
):
    """Entraîne un Random Forest (hyperparamètres du notebook).

    Args:
        X_train (DataFrame): features préparées.
        y_train (Series): cible binaire.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test.

    Args:
        model: modèle entraîné (doit avoir predict et predict_proba).
        X_test (DataFrame): features de test.
        y_test (Series): vraies étiquettes.

    Returns:
        dict: Accuracy, Precision, Recall, F1, ROC-AUC et matrice de confusion.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Confusion matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_model(obj, path):
    """Sauvegarde un objet (modèle ou préprocesseur) avec joblib."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_model(path):
    """Charge un objet sauvegardé avec joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
