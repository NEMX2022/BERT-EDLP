import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存，按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn import metrics
from keras.optimizers import Adam, SGD
from keras.layers import Input, Conv1D, Dropout, Flatten, Dense, BatchNormalization, Activation, Bidirectional, LSTM
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.utils import to_categorical
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
print(1)