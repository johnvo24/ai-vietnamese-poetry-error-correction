import os
import time
import random
import torch
import configs as config
from utils.dataframe_helper import *
from utils.data_helper import *
from Jvai import GDrive

# Time
def delay(from_=0.5, to_=2.5):
  time.sleep(random.uniform(from_, to_))