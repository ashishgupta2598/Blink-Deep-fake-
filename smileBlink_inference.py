import argparse
import os
import requests
import json
import shutil
from random import randint

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

import tensorflow as tf
import argparse
import numpy as np
# import sys
# sys.path.append('..')
from solu_base import Solu
from blink_net import BlinkLRCN
from solver import Solver
import cv2
from py_utils import x_utils as ulib
import time
import traceback
tf.compat.v1.disable_eager_execution()


def download_file(url):
    local_filename = url.split('/')[-1].split('?')[0]
    local_filename = local_filename.lower()
    random_string = str(randint(0, 10000000))
    org_filename ,file_extension = os.path.splitext(local_filename)
    new_filename = org_filename+random_string+file_extension

    r = requests.get(url, stream=True)

    directory = os.path.join(os.getcwd(),"tmp")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=os.path.join(os.getcwd(),"tmp/"+new_filename)
    with open(filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return filename



def return_prediction_video(blinkSmileStreamer,url):

    result = model_predict(blinkSmileStreamer, url)
    return result


def model_predict(blinkSmileStreamer, url):
    #folder_path = os.path.join('tmp', url.split('/')[-1].split('.')[0])
    time1 = time.time()
    ######DOWNLOAD VID
    input_vid_path = url
    out = blinkSmileStreamer.predict([input_vid_path])
    print(out)
    out  = out[0]
    print("time is",time.time()-time1)
    
    if out>=1:
        return "Yes",0
    return "No",0    
            

