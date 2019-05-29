echo "Hi. You are so charming and handsome."
echo
echo -n "Did you (git) commit [y/n]?"
read answer
echo

if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Well done!"
    echo
else
    echo "Thank you for committing responsibly."
    exit 1
fi

COMMIT=`git rev-parse HEAD`
NOW=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="deep_image_matting_${NOW}_${COMMIT}"
BUCKET="gs://secret-compass-237117-mlengine-us-west-1"

echo "Submitting $JOB_NAME"

gcloud ai-platform jobs submit training $JOB_NAME \
	--staging-bucket $BUCKET \
        --package-path trainer \
        --module-name trainer.task \
        --job-dir $BUCKET/$JOB_NAME \
        --region us-west1 \
	--python-version 3.5 \
	--runtime-version 1.13 \
        --config gcloud/config.yaml \
        --stream-logs \
	-- \
	--stage encoder_decoder
