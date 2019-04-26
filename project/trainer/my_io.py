from urllib.parse import urlparse
from google.cloud import storage
import cv2 as cv
import numpy as np
from tensorflow.python.lib.io import file_io
import os

def is_gcloud(url_str):
    return urlparse(url_str).scheme == 'gs'

def __get_blob(url_str):
    url = urlparse(url_str)
    bucket_name = url.netloc
    path = url.path[1:] # remove leading '/'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    return bucket.get_blob(path)

def __download_as_string(url_str):
    print("Downloading '%s'..." % url_str)
    ret = __get_blob(url_str).download_as_string()
    print("Downloaded '%s'" % url_str)
    return ret

def __download_as_file(url_str, dest):
    print("Downloading '%s' to '%s'" % (url_str, dest))
    __get_blob(url_str).download_to_filename(dest)
    print("Downloaded '%s'" % dest)
    return

def imread(url_str, flags=1, cache_dir=None):
    if is_gcloud(url_str):
        if cache_dir:
            filename = os.path.join(cache_dir, url_str.split('/')[-1])
            if not os.path.isfile(filename):
                os.makedirs(cache_dir, exist_ok=True)
                __download_as_file(url_str, filename)
            return cv.imread(filename, flags)
        else:
            im_bytes = bytearray(__download_as_string(url_str))
            im_np = np.asarray(im_bytes, dtype=np.uint8)
            return cv.imdecode(im_np, flags)
    else:
        # local file
        return cv.imread(url_str, flags)

def read_lines(url_str):
    if is_gcloud(url_str):
        data = __download_as_string(url_str)
        return data.decode('utf-8').splitlines()
    else:
        with open(url_str) as f:
            return f.read().splitlines()

def cache(from_url, cache_to):
    if os.path.isfile(cache_to):
        return

    os.makedirs(os.path.dirname(cache_to), exist_ok=True)

    print("Downloading: '%s' to '%s'" % (from_url, cache_to))
    __file_to_file(from_url, cache_to)
    print("Downloaded")

def save_to_gcloud(from_path, to_url):
    if not is_gcloud(to_url):
        raise ValueError("Expected gcloud url, got %s" % to_url)

    print('Uploading: %s' % to_url)
    __file_to_file(from_path, to_url)
    print('Uploaded')

def __file_to_file(from_path, to_path):
    with file_io.FileIO(from_path, mode='rb') as in_file:
        with file_io.FileIO(to_path, mode='wb+') as out_file:
            for chunk in iter(lambda: in_file.read(4096 * 32), b''):
                out_file.write(chunk)

if __name__ == '__main__':
    cv.imshow('image', imread('gs://secret-compass-237117-mlengine/data/fg/0cdf5b5d0ce1_01.jpg', cache_dir='cache/fg'))
    cv.waitKey(0)
    cv.destroyAllWindows()
