# -*- coding: utf-8 -*-
"""
@author: Marta Basquens

Set all settings for main program. 

Usage: 
    1. Set all desired settings prior running "Main" script. 

For more information refer to the Bachelor Thesis
"STUDY OF THE IMPLEMENTATION OF AN AUTONOMOUS DRIVING SYSTEM"
"""

import carla

# 1. Determine the connection Python - CARLA simulator #######################
# HOST SETTINGS 
HOST = 'localhost'
PORT = 2000
TIMEOUT = 5.0

# 2. Determine some visualization parameters #################################

# SIMULATOR  ------------------------
CAR_MODEL = "model3"
FPS = 60
# CAMERA FORMAT ---------------------
IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = True 
            # If true, a window with IM_WIDTH x IM_HEIGHT dimensions with first 
            # person car image will be shown. 
            # Caution: it slows down the performance of the training. 
            
# SPAWN POINT -------------------------
FIXED_SPAWN = True
SPAWN_POINT = carla.Location(x=96, y=39.61, z=15.4) 
ROTATION = carla.Rotation(yaw=15)

# 3. Determine the type of usage of the program ##############################
TRAINING = True
MODEL_NAME = "base_model"  
###
LOAD_MODEL = False
MODEL = './models/' 
# If the model is loaded,fill in the following:
AVG_REWARD = -50.0
MIN_REWARD = -100.0
PRV_STEP = 0

# 4. Adjust the parameters of the model #####################################
# MODEL PARAMETERS
EPISODES = 7000
DISCOUNT = 0.95
epsilon = 1
EPSILON_DECAY = 0.999975
EPSILON_RECOVERY = 1.00000025
MIN_EPSILON = 0.35
MAX_EPSILON = 0.55

MEMORY_FRACTION = 0.4

# REINFORCEMENT LEARNING PARAMETERS
AGGREGATE_STATS_EVERY = 10
REPLAY_MEMORY_SIZE = 3_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

# 5. Configure the simulator with all the desired sensors ###################
CAMERA = True
SEMANTIC_SEGMENTATION = True # If false, RGB camera is used instead
LIDAR = False
GNSS = False
LANE_INVASION = False
COLLISION_SENSOR = True 