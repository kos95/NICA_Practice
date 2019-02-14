import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets

import numpy as np
import skimage
from scipy.misc import toimage

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tifffile import imsave
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction import image

from PIL import Image
from torchvision.transforms import ToTensor

import csv
from random import randint

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = False

##cuda check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_PATH = r'F:\Research Database\Deep_Neural_Networks_Segment_Neuronal_Membranes_in_Electorn_Microscopy_Images'
PATCH_PATH = './image_patch'
MODEL_STORE_PATH = r'F:\Research Database\Deep_Neural_Networks_Segment_Neuronal_Membranes_in_Electorn_Microscopy_Images\pytorch_models'

train_image = skimage.io.imread(os.path.join(DATA_PATH, 'train-volume.tif'), plugin='tifffile')
label_image = skimage.io.imread(os.path.join(DATA_PATH, 'train-labels.tif'), plugin='tifffile')
test_image = skimage.io.imread(os.path.join(DATA_PATH, 'test-volume.tif'), plugin='tifffile')

window_size = 95
n_picture = 30
new_dataset = 0
new_test_dataset = 0
if (new_dataset == 1):
    f = open('data_csv.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for pic in range(n_picture):
        patches_train = image.extract_patches_2d(train_image[pic], (window_size, window_size))
        patches_label = image.extract_patches_2d(label_image[pic], (window_size, window_size))
        for i in range(1000):
            rand_int = randint(0, patches_train.shape[0]-1)
            patch_filename = "./image_patch/" + str(pic*1000+i) + ".tif"
            imsave(patch_filename, patches_train[rand_int])
            wr.writerow([pic*1000+i, int(patches_label[rand_int,int((window_size-1)/2),int((window_size-1)/2)]/255)])
    f.close()

if (new_test_dataset == 1):
    f = open('data_csv_test.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for pic in range(1):
        patches_test = image.extract_patches_2d(test_image[pic], (window_size, window_size))
        rng = int(patches_test.shape[0]-1)
        for i in range(rng):
            patch_filename = "./image_patch_test/" + str(i) + ".tif"
            imsave(patch_filename, patches_test[pic*1000+i])
            wr.writerow([pic*1000+i])
    f.close()


class MemDataset(torch.utils.data.Dataset):
    def __init__(self):
        xy = np.loadtxt('data_csv.csv', delimiter=',', dtype = np.int64)
        self.length = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        b = self.x_data[index].numpy()[0]

        patch_filename = "./image_patch/" + str(b) + ".tif"
        image = Image.open(patch_filename)
        image = self.transform(image)
        label = self.y_data[index]
        return image, label
    def __len__(self):
        return self.length


class MemDataset_test(torch.utils.data.Dataset):
    def __init__(self):
        patches_test = image.extract_patches_2d(test_image[0], (window_size, window_size))
        self.length = patches_test.shape[0]
        self.x_data = patches_test
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        image = self.transform(toimage(self.x_data[index]))
        return image
    def __len__(self):
        return self.length

# Hyperparameters
num_epochs = 1
num_classes = 2
batch_size = 200
learning_rate = 0.001

train_dataset = MemDataset()
test_dataset = MemDataset_test()
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

'''
labels_map = {0 : 'mem', 1 : 'non-mem'};

fig = plt.figure(figsize=(8,8));
columns = 4;
rows = 5;
print(train_dataset)
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_dataset));
    img = train_dataset[img_xy][0]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[int(train_dataset[img_xy][1].numpy()[0])])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()
'''



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(432, 200)
        self.fc2 = nn.Linear(200, 2)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


model = ConvNet()

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
       
        labels = labels.squeeze_()
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

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Test the model
model.eval()

result_list = []
f = open('data_result.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
patches_test = image.extract_patches_2d(test_image[0], (window_size, window_size))
total_len = patches_test.shape[0]
wr = csv.writer(f)
with torch.no_grad():
    correct = 0
    total = 0

    for images in test_loader:
        images= images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        result_list.append(predicted)
        wr.writerow([predicted])
f.close()

