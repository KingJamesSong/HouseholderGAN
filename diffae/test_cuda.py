import os

gpu_info = os.popen('nvcc --version').read()
print(gpu_info)