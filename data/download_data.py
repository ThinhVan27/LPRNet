# The following code will only execute
# successfully when compression is complete

import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("thinhvan/license-plate-recognition")

print("Path to dataset files:", path)

shutil.copytree(path)
