TRAINER_PACKAGE_PATH="trainer"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="deep_image_matting_$now"
MAIN_TRAINER_MODULE="trainer.task"
JOB_DIR="gs://secret-compass-237117-mlengine/models"
PACKAGE_STAGING_PATH="gs://secret-compass-237117-mlengine"

gcloud ai-platform jobs submit training $JOB_NAME \
	--staging-bucket $PACKAGE_STAGING_PATH \
        --package-path $TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        --job-dir $JOB_DIR \
        --region us-central1 \
	--python-version 3.5 \
	--runtime-version 1.13 \
        --config gcloud/config.yaml
