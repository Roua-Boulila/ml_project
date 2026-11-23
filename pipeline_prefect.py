"""
pipeline_prefect.py : orchestration du pipeline ML avec Prefect.
Chaque task appelle une fonction définie dans main.py.

Usage :
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
"""

import argparse
import os
import subprocess  # nosec B404
import sys

from prefect import flow, get_run_logger, task

from main import step_evaluate, step_load, step_prepare, step_save, step_train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/Roua-Boulila/ml_project.git"
PROJECT_DIR = os.path.expanduser("~/ml_project")

DATA_PATH = os.path.join(BASE_DIR, "adult.data")
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd, cwd=None) -> None:
    """Exécute une commande et lève une erreur si elle échoue."""
    subprocess.run(cmd, check=True, cwd=cwd)  # nosec B603 B607


# ------------------------------- TASKS -------------------------------
@task(name="fetch-project")
def fetch_project() -> None:
    """Récupère le projet depuis le repo Git distant (clone ou pull)."""
    logger = get_run_logger()
    if os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
        run_cmd(["git", "-C", PROJECT_DIR, "pull", "--ff-only", "origin", "main"])
        logger.info("Project updated (git pull).")
    else:
        run_cmd(["git", "clone", REPO_URL, PROJECT_DIR])
        logger.info("Project cloned (git clone).")


@task(name="install-dependencies")
def install_dependencies() -> None:
    """1) Installer les dépendances."""
    logger = get_run_logger()
    run_cmd(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=BASE_DIR,
    )
    logger.info("Dependencies installed.")


@task(name="format-code")
def format_code() -> None:
    """Formatage du code avec black."""
    logger = get_run_logger()
    run_cmd([sys.executable, "-m", "black", *CODE_FILES], cwd=BASE_DIR)
    logger.info("Code formatted (black).")


@task(name="code-quality")
def code_quality() -> None:
    """Qualité du code avec flake8."""
    logger = get_run_logger()
    run_cmd(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *CODE_FILES],
        cwd=BASE_DIR,
    )
    logger.info("Code quality OK (flake8).")


@task(name="code-security")
def code_security() -> None:
    """Sécurité du code avec bandit."""
    logger = get_run_logger()
    run_cmd([sys.executable, "-m", "bandit", "-q", *CODE_FILES], cwd=BASE_DIR)
    logger.info("Code security OK (bandit).")


@task(name="unit-tests")
def unit_tests() -> None:
    """Exécution des tests unitaires avec pytest."""
    logger = get_run_logger()
    run_cmd([sys.executable, "-m", "pytest", "test_pipeline.py", "-v"], cwd=BASE_DIR)
    logger.info("Unit tests passed (pytest).")


@task(name="prepare-data")
def prepare_data_task():
    """2) Préparation des données."""
    logger = get_run_logger()
    result = step_prepare(DATA_PATH)
    logger.info("Data preparation complete.")
    return result


@task(name="train-model")
def train_model_task(X_train, y_train):
    """3) Entraînement du modèle."""
    logger = get_run_logger()
    model = step_train(X_train, y_train)
    logger.info("Model training complete.")
    return model


@task(name="save-model")
def save_model_task(model, preprocessor) -> None:
    """4) Sauvegarder le modèle."""
    logger = get_run_logger()
    step_save(model, preprocessor, MODEL_PATH, PREPROCESSOR_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")


@task(name="load-model")
def load_model_task():
    """5) Charger le modèle."""
    logger = get_run_logger()
    model = step_load(MODEL_PATH)
    logger.info(f"Model loaded: {MODEL_PATH}")
    return model


@task(name="evaluate-model")
def evaluate_model_task(model, X_test, y_test):
    """6) Évaluer le modèle."""
    logger = get_run_logger()
    metrics = step_evaluate(model, X_test, y_test)
    logger.info("Model evaluation complete.")
    return metrics


# ------------------------------- FLOWS -------------------------------
@flow(name="code", log_prints=True)
def flow_code() -> None:
    """Récupération, dépendances, formatage, qualité, sécurité, tests."""
    fetch_project()
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="all", log_prints=True)
def flow_all() -> None:
    """Code (fetch, install, format, qualité, sécurité, tests) puis 2-3-4-6."""
    fetch_project()
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model, preprocessor)
    evaluate_model_task(model, X_test, y_test)


@flow(name="train", log_prints=True)
def flow_train() -> None:
    """Préparation des données + entraînement du modèle."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_task()
    train_model_task(X_train, y_train)


@flow(name="evaluate", log_prints=True)
def flow_evaluate() -> None:
    """Chargement du modèle + évaluation."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, X_test, y_test)


FLOWS = {
    "code": flow_code,
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
}


# ------------------------------- CLI --------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), required=True)
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
