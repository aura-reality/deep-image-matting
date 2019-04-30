now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="deep_image_matting_$now"
BUCKET="gs://secret-compass-237117-mlengine"

gcloud ai-platform jobs submit training $JOB_NAME \
	--staging-bucket $BUCKET \
        --package-path trainer \
        --module-name trainer.task \
        --job-dir $BUCKET/$JOB_NAME \
        --region us-west1 \
	--python-version 3.5 \
	--runtime-version 1.13 \
        --config gcloud/config.yaml \
        --stream-logs
