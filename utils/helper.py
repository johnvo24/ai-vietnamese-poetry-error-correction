import time
import random
from utils.dataframe_helper import *
from utils.data_helper import *
from utils.adaptive_random import AdaptiveRandom

# Time
def delay(from_=0.5, to_=2.5):
  time.sleep(random.uniform(from_, to_))
