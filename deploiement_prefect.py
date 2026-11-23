"""
deploiement_prefect.py : déploiement des flows Prefect.
- 'all' est planifié tous les jours à 02:00 (heure de Tunis)
- les autres flows sont déclenchables manuellement
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_code, flow_evaluate, flow_train

if __name__ == "__main__":
    serve(
        flow_all.to_deployment(
            name="ml-pipeline-all",
            schedule=CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis"),
            tags=["mlops", "full-pipeline"],
        ),
        flow_train.to_deployment(
            name="ml-pipeline-train",
            tags=["mlops", "training"],
        ),
        flow_evaluate.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["mlops", "evaluation"],
        ),
        flow_code.to_deployment(
            name="ml-pipeline-code",
            tags=["mlops", "code-quality"],
        ),
    )
