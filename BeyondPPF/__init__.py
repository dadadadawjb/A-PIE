from .models.model import create_shot_encoder, create_encoder

# due to inference is not a module
import importlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src_shot.build
import utils
import models.model
import models.voting
importlib.reload(src_shot.build)
importlib.reload(utils)
importlib.reload(models.model)
importlib.reload(models.voting)
from .inference import inference
sys.path.remove(os.path.dirname(os.path.abspath(__file__)))
