import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform


def check_gpu_arm() -> None:
    """
    Check the GPU and ARM version of the system.

    All the code is from: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install
    -mac-metal-jan-2023.ipynb

    :return:
    """
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
