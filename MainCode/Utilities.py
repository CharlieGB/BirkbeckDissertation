import os
import numpy as np
import shutil

def InitialiseGPU():
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # No GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="," # No GPU / PyTorch


def RecordSettings(directory, maze_params, agent_params):
    file = open(directory + 'Settings.txt', 'w')

    for key, value in maze_params.items():
        file.write(key + ': ' + str(value) + '\n')

    for key, value in agent_params.items():
        file.write(key + ': ' + str(value) + '\n')

    file.close()
    return

def RecordSetting(directory, key, value):
    file = open(directory + 'Settings.txt', 'a')
    file.write(key + ': ' + str(value) + '\n')
    file.close()
    return
