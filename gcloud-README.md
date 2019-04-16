## Copy data to google cloud
Make sure all your data is in the `/data` folder
Then, you can Try to run `./copy-to-gcloud-bucket.sh`
It intermittently works for me; I get lots of retries (maybe due to bad internet).
If the scripts doesn't work, you can always upload the files through the console.
Wasn't able to get the command line working -- do it from the browser.

## Download credentials
To interact with google cloud with python, download credentials.
See the section titled 'Setting up authentication'
https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python
