import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
N_EPOCHS = 10
NUM_WORKERS = 2
SAVE_MODEL = True
LOAD_MODEL = False
CURR_DIR = os.path.abspath(os.path.basename(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURR_DIR, "..", "data"))
NDATA_DIR = os.path.abspath(os.path.join(CURR_DIR, "..", "norm_data"))