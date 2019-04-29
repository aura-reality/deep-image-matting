gcloud ai-platform local train \
	--package-path trainer \
	--module-name trainer.task \
	--job-dir ../models \
	-- \
	--stage encoder_decoder
