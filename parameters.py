import torch
import numpy as np
import os

_CUR_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\','/') + '/'

###############################################################################
# hardware mode
###############################################################################

DEVICE_CPU = torch.device("cpu")
DEVICE_GPU = torch.device("cuda:0")
DEVICE = DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU
NUMPY_DTYPE = np.float32
TORCH_DTYPE = torch.float32
NUMPY_DTYPE_BIN = np.int
TORCH_DTYPE_BIN = torch.float32

PARALLEL = True
WORKERS = 6
if(not PARALLEL): WORKERS = 0

###############################################################################
# Machine learning training parameters
###############################################################################

# parameters
MODEL_MODE = "train" # "train"/"valid"/"test"
NEW_NORM = True
VALIDATION = True
NORM_PERC = 1.00
MODEL_NUM = -1
TRAIN_STAGE = 1 # 1:stage1, 2:stage2

LEARNING_RATE = 1e-4
NOISE = 0.10
NOISE_METHOD = "SD" # "SD", "PERC"
BATCH_SIZE = 64
MAX_EPOCH = 20
SAVE_INTERVAL = 3

STAGE_ONE_MAX_EPOCH = 10
STAGE_TWO_MAX_EPOCH = 100

LOG_OUT_INTERVAL = 10

###############################################################################
# Machine learning model parameters
###############################################################################

EDGE_INT_VEC_LEN = 128
NODE_INT_VEC_LEN = 64
EDGE_CODE_LEN = 128
NODE_CODE_LEN = 64

WIDTH_FACTOR = 1
STAGE_ONE_WIDTH = 1
STAGE_TWO_WIDTH = 0.5

USE_PCA = True
PCA_ALL = False
PCA_BC = False
PCA_THRESHOLD = 0.98
POST_PCA_MEAN_VAR = True

USE_PERC = True # False to use std
VAR_PERCENTILE = 95

DOUBLE_GN = True
USE_DELTA = True

CAL_DISLOCATION = True
DISLOC_GRAD_SCALE = 1.0
PROP_DISLOCATION = DISLOC_GRAD_SCALE > 0
TARGET_GRAD_SCALE = 1.0

CUT_LAST = False

SIM_INCREMENT = 10

###############################################################################
# Data extraction parameters
###############################################################################

EXTRACT = True
COMPILE = True

CUT_DATA = 500

SAMP_STRESS = True
JOINT_SAMP_STRESS = False
SAMP_CORNER = True
SAMP_FIVE_FRAMES = False

USE_MIRROR = True

TWO_STAGE = True
STAGE_TWO_INCREMENT = 1
STAGE_TWO_FREQ = 1

FRAME_FREQ = int(10 / SIM_INCREMENT)
TRAIN_RATIO = 0.8
TEST_FILE = _CUR_DIR + "test.json"
TRAIN_FILE = _CUR_DIR + "train.json"

###############################################################################
# Machine learning forward parameters
###############################################################################

INPUT_NAME = _CUR_DIR + "tests/models/joinery.json"
OUTPUT_NAME = _CUR_DIR + "tests/final/joinery.json"

STAGE_ONE_MODEL = _CUR_DIR + "selected_models/stage_1/model.pth"
STAGE_ONE_NORM =  _CUR_DIR + "selected_models/stage_1/normalizer.json"
STAGE_TWO_MODEL = _CUR_DIR + "selected_models/stage_2/model.pth"
STAGE_TWO_NORM =  _CUR_DIR + "selected_models/stage_2/normalizer.json"
INFO_FILE = _CUR_DIR + "selected_models/info.json"

STAGE_ONE_MODEL_SHAPE = [512, 512, 512, 256]
STAGE_ONE_MODEL_WIDTH = 1
STAGE_ONE_NODE_INT_VEC_LEN = 64
STAGE_ONE_EDGE_INT_VEC_LEN = 128

STAGE_TWO_MODEL_SHAPE = [512, 512, 512, 256]
STAGE_TWO_MODEL_WIDTH = 1
STAGE_TWO_NODE_INT_VEC_LEN = 64
STAGE_TWO_EDGE_INT_VEC_LEN = 64

# CONDUIT_PATH = "C:/CHI_design_tool/conduit.json"

###############################################################################
# FEA raw data paths
###############################################################################

INP_FOLDER = "./input_files"
FIL_FOLDER = "./node_files"
EXT_FOLDER = "./extracted_files"
MODEL_FOLDER = "./models"
DATA_FOLDER = "./data"
REF_OUT_FOLDER = "./ref_models"
STAGE_TWO_FOLDER = "/stage2"

###############################################################################
# Constants
###############################################################################

EPSILON = 0.000001
LINE_LEN = 10 # int, maximum number of elements in a line in the input file
COOR_LEN = 3 # int, coordinate vector length (x, y, z)
STRESS_LEN = 6 # int, stress value length [x, y, z, x, y, z]
                                         # (load)  (shear)
GP_LEN = 8 # int, gaussian point count per element
GP_MAP = {0:0, 1:1, 2:3, 3:2, 4:4, 5:5, 6:7, 7:6}
ROT_PER_TRIAL = 10

###############################################################################
# Indexing parameters
###############################################################################

INDEX_LEN = 3 # digits to use in index coords
BEAM_INDEX = 1 # int to use for beams in IDString
JOINT_INDEX = 2 # int to use for joints in IDString
NODE_INDEX = 0 # int to use for nodes in IDString
ELEM_INDEX = 1 # int to usd for elements in IDString
NODE_SET_INDEX = 2 # int to usd for node sets in IDString
ELEM_SET_INDEX = 3 # int to usd for element sets in IDString

###############################################################################
# Tokens
###############################################################################
# tokens to look for in the input file

# model design
MODEL_START_TOKEN = "****Model design start"
MODEL_END_TOKEN = "****Model design end"
JOINT_START_TOKEN = "**Joints"
BEAM_START_TOKEN = "**Beams"
# nodes
NODE_MAP_START_TOKEN = "****Node index map start"
NODE_MAP_END_TOKEN = "****Node index map end"
NODE_START_TOKEN = "****Node code start"
NODE_END_TOKEN = "****Node code end"
# elements
ELEM_MAP_START_TOKEN = "****Element index map start"
ELEM_MAP_END_TOKEN = "****Element index map end"
ELEM_START_TOKEN = "****Element code start"
ELEM_END_TOLKEN = "****Element code end"
# elsets
ELEM_SET_START_TOLKEN = "****Elset code start"
ELEM_SET_END_TOKEN = "****Elset code end"
# nsets
NODE_SET_START_TOKEN = "****Nset code start"
NODE_SET_END_TOKEN = "****Nset code end"
# boundary conditions
BC_START_TOKEN = "****BC code start"
BC_END_TOKEN = "****BC code end"
# solver configurations
INCREMENT_TOKEN = "inc="
FREQ_TOKEN = "*node file, frequency="

# extracted data output parameters
INP_SUFFIX = ".inp"
COOR_FILE_SUFFIX = "_coor.npy"
STRESS_FILE_SUFFIX = "_stress.npy"
JSON_SUFFIX = ".json"
TEXT_SUFFIX = ".txt"
CSV_SUFFIX = ".csv"
NUMPY_SUFFIX = ".npy"
ALL_TOKEN = "all"
NORM_TOKEN = "normalizer"
MIRROR_TOKEN = "_m"
# block tokens
BEAM_TOKEN = "_beam"
JOINT_TOKEN = "_joint"
# dataset tokens
NUMPY_SUFFIX = ".npy"
INFO_TOKEN = "info"
STAGE_TWO_TOKEN = "stage2"
ADJ_MATRIX_TOKEN = "adjMat"
NODES_TOKEN = "nodes"
EDGES_TOKEN = "edges"
NODES_TAR_TOKEN = "nodes_target"
EDGES_TAR_TOKEN = "edges_target"
if(USE_DELTA):
    NODES_TAR_TOKEN = "nodes_delta"
    EDGES_TAR_TOKEN = "edges_delta"
EDGE_NUMTYPE_TOKEN = "edgeNumType"
NODE_NUMTYPE_TOKEN = "nodeNumType"
ROT_ANGS_TOKEN = "rot_angs"
CUMSUM_TOKEN = "_cumsum"

# data utility tokens
POINT_TOKEN = 'p'
VECTOR_TOKEN = 'v'
STRESS_TOKEN = 's'
BC_TOKEN = 'b'
EDGE_CEN_TOKEN = "edgeCenter"
NODE_CEN_TOKEN = "nodeCenter"
ORDER_TOKEN = "order"

# machine learning tokens
TRAIN_TOKEN = "train"
TEST_TOKEN = "test"
FORWARD_TOKEN = "forward"
LOG_TOKEN = "log"

###############################################################################
# composite names
###############################################################################
# tokens to look for in the input file

NORMALIZER_FILENAME = NORM_TOKEN + JSON_SUFFIX
ORDER_FILENAME = DATA_FOLDER + '/' + ORDER_TOKEN + JSON_SUFFIX
LOG_FILENAME = LOG_TOKEN + TEXT_SUFFIX

###############################################################################
# processing
###############################################################################

LEARNING_RATE *= (64 / BATCH_SIZE) # using 64 as baseline