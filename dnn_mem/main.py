import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np

from skimage import io
from matplotlib import pyplot as plt

from PIL import Image
from torchvision.transforms import ToTensor


torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = False

##cuda check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_PATH = r'F:\Research Database\Deep_Neural_Networks_Segment_Neuronal_Membranes_in_Electorn_Microscopy_Images'
MODEL_STORE_PATH = r'F:\Research Database\Deep_Neural_Networks_Segment_Neuronal_Membranes_in_Electorn_Microscopy_Images\pytorch_models'

train_image = io.imread(os.path.join(DATA_PATH, 'train-volume.tif'))
label_image = io.imread(os.path.join(DATA_PATH, 'train-labels.tif'))
test_image = io.imread(os.path.join(DATA_PATH, 'test-volume.tif'))


def extract_patches_2D(img, size):



class N4model(nn.Module):
    def __init__(self):
        super(N4model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=4,stride = 2, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2, padding=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 5, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2, padding=0),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 4, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2, padding=0),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 4, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2, padding=0),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out



model = N4model


model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()