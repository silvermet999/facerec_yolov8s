import ultralytics
from ultralytics import YOLO
import os
from IPython.display import clear_output, Image
from pathlib import Path
import subprocess

from torch import cuda
from torch.cuda import empty_cache
import torchvision

clear_output()
cuda.synchronize()
ultralytics.checks()
import torch


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda" if torch.cuda.is_available() else 'cpu'
print({device})
model = YOLO("yolov8s.yaml").to(device)
results = model.train(data="datasets/data.yaml", epochs=10)
torch.save(model.state_dict(), 'yolov8s.pt')
eval = model.val()
img = Path("datasets/valid/images/0_10725_jpg.rf.86a1472efaea6984d3f4fa02b0bc88ec.jpg")

# command = "yolo task=detect mode=train model=yolov8s.yaml data=dataset/data.yaml epochs=10 imgsz=60"
# try:
#     subprocess.run(command, shell=True, check=True)
# except subprocess.CalledProcessError as e:
#     print("Error executing command:", e)

empty_cache()