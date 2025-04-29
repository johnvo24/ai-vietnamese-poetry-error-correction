import os

# ================================
# TRAIN SETUP
USE_GDRIVE=False
MAX_REASONING_MEMORY = 6 # n steps = step 0 & n-1 last steps
MAX_INPUT_LENGTH = 512 # DeepSeekVpec->512
MAX_LENGTH = 1024 # GPT2-256~512, DeepSeekVpec-512~4096->1024

# MODEL
VPEC_GPT2_MODEL_NAME = 'vpec_gpt2'
VPEC_DEEPSEEK_MODEL_NAME = 'vpec_deepseek'

# SFT TRAINING
# Dataset
SFT_DATASET_FILE_PATH = 'data/sft_dataset/poetryfix_gold_data.csv'
SFT_TRAIN_SIZE = 80 # %
SFT_VAL_SIZE = 10 # %
SFT_TEST_SIZE = 10  # %
SFT_RANDOM_STATE = 42
# Dataset loader
BATCH_SIZE = 2
NUM_WORKERS = 4 # Num of subprocesses to use for data loading
SHUFFLE = False
# Training
SFT_VPECGPT2_EPOCHS = 100
SFT_VPECGPT2_LEARNING_RATE = 03e-5
SFT_VPECDEEPSEEK_EPOCHS = 100
SFT_VPECDEEPSEEK_LEARNING_RATE = 03e-5


# _note_ > Path variable configurations
DATA_DIR = os.path.join(os.getcwd(), 'data')
SFT_DATASET_DIR = os.path.join(DATA_DIR, 'sft_dataset')
SRC_DIR = os.path.join(os.getcwd(), 'src')
OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# =================================
SFT_TRAIN_DATASET_PATH = os.path.join(SFT_DATASET_DIR, 'train_dataset.csv')
SFT_VAL_DATASET_PATH = os.path.join(SFT_DATASET_DIR, 'val_dataset.csv')
SFT_TEST_DATASET_PATH = os.path.join(SFT_DATASET_DIR, 'test_dataset.csv')