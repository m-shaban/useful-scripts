"""
Task: Classify a given image using ImageNet based pretrained image classification model (e.g., Resnet101)
"""

import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# path of required files
img_path = 'dog.png'
imagenet_labels_file_path = 'imagenet_classes.txt'

# input data and imagenet labels
img = Image.open(img_path)
with open(imagenet_labels_file_path) as f:
    labels = [line.strip() for line in f.readlines()]

# preprocessor and model initialization
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
resnet = models.resnet101(pretrained=True)
resnet.eval()

# model inference
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
out = resnet(batch_t)
_, index = torch.max(out, 1)
_, indicies = torch.sort(out, 1, descending=True)
probabilities = torch.nn.functional.softmax(out, 1)[0]

# results
print('Top-1 Prediction Label: %s' % labels[index[0]])
print('Top-1 Prediction Confidense: %0.04f' % probabilities[index[0]])
print('')
print('Top-5 Predictions with Confidence Values:')
i = 1
for idx in indicies[0][:5]:
    print("%d - %s: %0.04f" % (i, labels[idx], probabilities[idx]))
    i+= 1
