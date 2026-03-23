# The following code will only execute
# successfully when compression is complete

import kagglehub
import shutil
import os
import random

random = random.seed(42)

# Download latest version
path = kagglehub.dataset_download("thinhvan/lpr-dataset")

print("Path to dataset files:", path)

current_dir = os.path.dirname(os.path.abspath(__file__))
shutil.move(os.path.join(path, "data", "train"), current_dir)
shutil.move(os.path.join(path, "data", "test"), current_dir)


            