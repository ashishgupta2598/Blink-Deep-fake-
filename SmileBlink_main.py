import os,sys,time
import argparse
import falcon, json
import ntpath
from random import randint
import os, shutil
from random import randint
import requests
from PIL import Image
import smileBlink_inference
import threading
import sys
import traceback
from loguru import logger
logger.add(sys.stderr, colorize=True, format="<green>{time}</green> {level} {message}", filter="my_module", level="INFO", backtrace=True)

from smileBlink_model import BlinkSmile
smileBlinkStreamer=BlinkSmile()#None


video_url = "0001.mp4"
smile,blink = smileBlink_inference.return_prediction_video(smileBlinkStreamer, video_url)
print(smile,blink)