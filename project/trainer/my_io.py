from urllib.parse import urlparse
from google.cloud import storage
import cv2 as cv
import numpy as np
from tensorflow.python.lib.io import file_io
import os

def __is_gcloud(url_str):
    return urlparse(url_str).scheme == 'gs'

def __get_blob(url_str):
    url = urlparse(url_str)
    bucket_name = url.netloc
    path = url.path[1:] # remove leading '/'

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    print('Downloading: %s' % url_str)
    return bucket.get_blob(path)

def __download_as_string(url_str):
    return __get_blob(url_str).download_as_string()

def imread(url_str, flags=1):
    if __is_gcloud(url_str):
        im_bytes = bytearray(__download_as_string(url_str))
        im_np = np.asarray(im_bytes, dtype=np.uint8)
        return cv.imdecode(im_np, flags)
    else:
        # local file
        return cv.imread(url_str, flags)

def read_lines(url_str):
    if __is_gcloud(url_str):
        data = __download_as_string(url_str)
        return data.decode('utf-8').splitlines()
    else:
        with open(url_str) as f:
            return f.read().splitlines()

def cache(from_url, cache_to):
    if not os.path.isfile(cache_to):
        the_file = file_io.FileIO(from_url, mode='rb')
        temp_model_location = cache_to
        temp_the_file = open(temp_model_location, 'wb')
        print('Downloading: %s' % from_url)
        temp_the_file.write(the_file.read())
        temp_the_file.close()
        the_file.close()
