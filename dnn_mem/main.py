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

from bokeh.plotting import figure
from bokeh.io import show, save, output_file
from bokeh.models import LinearAxis, Range1d

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = False

##cuda check
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_PATH = r'F:\Research Database\Deep_Neural_Networks_Segment_Neuronal_Membranes_in_Electorn_Microscopy_Images'
PATCH_PATH = "./image_patch_3000/"
MODEL_STORE_PATH = r'F:\Research Database\Deep_Neural_Networks_Segment_Neuronal_Membranes_in_Electorn_Microscopy_Images\pytorch_models'

train_image = skimage.io.imread(os.path.join(DATA_PATH, 'train-volume.tif'), plugin='tifffile')
label_image = skimage.io.imread(os.path.join(DATA_PATH, 'train-labels.tif'), plugin='tifffile')
test_image = skimage.io.imread(os.path.join(DATA_PATH, 'test-volume.tif'), plugin='tifffile')


# Hyperparameters
patch_per_image = 3000
num_epochs = 150
num_classes = 2
batch_size = 50
learning_rate = 0.001

window_size = 95
n_picture = 30
new_dataset = 1
new_test_dataset = 0


if (new_dataset == 1):
    f = open('data_csv.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for pic in range(n_picture):
        patches_train = image.extract_patches_2d(train_image[pic], (window_size, window_size))
        patches_label = image.extract_patches_2d(label_image[pic], (window_size, window_size))
        for i in range(patch_per_image):
            rand_int = randint(0, patches_train.shape[0]-1)
            patch_filename = PATCH_PATH + str(pic*patch_per_image+i) + ".tif"
            imsave(patch_filename, patches_train[rand_int])
            wr.writerow([pic*patch_per_image+i, int(patches_label[rand_int,int((window_size-1)/2),int((window_size-1)/2)]/255)])
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

        patch_filename = PATCH_PATH + str(b) + ".tif"
        image = Image.open(patch_filename)
        image = self.transform(image)
        label = self.y_data[index]
        return image, label
    def __len__(self):
        return self.length


class MemDataset_test(torch.utils.data.Dataset):
    def __init__(self):
        pad = int((window_size-1)/2)
        pad_img = np.pad(test_image[0],((pad,pad),(pad,pad)),'reflect')
        patches_test = image.extract_patches_2d(pad_img, (window_size, window_size))
        self.length = patches_test.shape[0]
        self.x_data = patches_test
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        image = self.transform(toimage(self.x_data[index]))
        return image
    def __len__(self):
        return self.length



train_dataset = MemDataset()
test_dataset = MemDataset_test()
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
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
    train_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)


        # Run the forward pass
        
        outputs = model(images)
       
        labels = labels.squeeze_()
        loss = criterion(outputs, labels)

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()


        train_loss += loss.item()
        total_train += total
        correct_train += correct

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    epoch_loss = train_loss / (i+1)
    epoch_acc = (correct_train/total_train)
    acc_list.append(epoch_acc)
    loss_list.append(epoch_loss)


    

# Test the model
model.eval()

result_list = []


patches_test = image.extract_patches_2d(test_image[0], (window_size, window_size))
total_len = patches_test.shape[0]

#model.load_state_dict(torch.load('model.pth'))
#model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images in test_loader:
        images= images.to(device)
        outputs = model(images)


        #softmax
        _, predicted = torch.max(outputs.data, 1)

        for i in predicted:
            result_list.append(i.item())


f = open('data_result.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([result_list])
f.close()


result_list = np.asarray(result_list)
test_label = np.reshape(result_list, (512,-1))
test_label_img = test_label*255
fig = plt.figure(num=1,figsize=(512, 512))
fig.add_subplot(1, 2, 1)
plt.imshow(test_image[0], cmap = 'gray')
fig.add_subplot(1, 2, 2)
plt.imshow(test_label_img, cmap='gray')

plt.figure(2)
plt.plot(acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure(3)
plt.plot(loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()


torch.save(model.state_dict(), './model.pth')