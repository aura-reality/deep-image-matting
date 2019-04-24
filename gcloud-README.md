## Copy data to google cloud
* Make sure all your data is in the `/data` folder (specifically the `fg_path`, `bg_path`, and `a_path` as given in [config.py](config.py))
  * Run `./prepare-to-train.py`
* Then, you can try to run `./copy-to-gcloud-bucket.sh`
  * You'll first have to install the command line tools, see the section titled **Setup Your Environment** [here](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction)
  * It intermittently works for me; I get lots of retries (maybe due to bad internet).
* **@Erez: recommended / easier option:** If the script doesn't work, you can always [upload the files through the browser]( https://cloud.google.com/storage/docs/uploading-objects).
  * Upload to `gs://secret-compass-237117-mlengine/data` (i.e. mirror it with your local directory)
* Send me your `config.py`

## Download credentials
* To interact with google cloud with python (i.e. to train), you may need to download credentials.
  * See the section titled 'Setting up authentication' [here](https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python)

## Submit a job to google cloud
```
cd project
./gcloud/submit-training.sh
```

## View the job through the console
From here, you can see stats and stop the job.

In the main menu (three lines):
* Artificial Intelligence => AI Platform => Jobs
