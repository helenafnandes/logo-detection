import torch
import torchvision

print(torch.cuda.is_available())

print(torch.__version__)
print(torchvision.__version__)

#torch.cuda.empty_cache()