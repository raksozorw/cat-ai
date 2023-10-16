# model.py

import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(pretrained=True) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load('cat_recognition_model4.pth')) 
model.eval()