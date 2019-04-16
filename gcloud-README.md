## Copy data to google cloud
Make sure all your data is in the `/data` folder (specifically the fg_path, bg_path, and a_path as given in config.py)
Then, you can try to run `./copy-to-gcloud-bucket.sh`
It intermittently works for me; I get lots of retries (maybe due to bad internet).
If the script doesn't work, you can always upload the files through the console.
See https://cloud.google.com/storage/docs/uploading-objects

## Download credentials
To interact with google cloud with python (i.e. to train), you may need to download credentials.
See the section titled 'Setting up authentication'
https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
