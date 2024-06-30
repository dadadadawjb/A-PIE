from .models.model import PointNet2
from .utils import inplace_relu

# due to inference is not a module
import importlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src_shot.build
import utils
import models.model
importlib.reload(src_shot.build)
importlib.reload(utils)
importlib.reload(models.model)
from .inference import inference
from .grad_inference import grad_inference
sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
