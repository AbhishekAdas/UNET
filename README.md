# UNET
import torch
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from torchvision import transforms
import glob
import cv2
import os
import tqdm
from google.colab.patches import cv2_imshow
import torch.nn as nn
import matplotlib.pyplot as plt

LABELS = ['airplane', 'animal', 'car', 'human', 'truck']
batch_size = 64

def visualize(image):
  cv2_imshow(image)


def read_imgfilepath_label(path):
  label_folder = os.listdir(path)
  label = []
  image_path = []
  images = []
  for l_folder in label_folder:
     pth = glob.glob(os.path.join(path, l_folder, '*.jpg'))
     '''for x in pth:
       image = cv2.imread(x) 
       #images.append(image) '''
     image_path.extend(pth)
     #print(LABELS.index(l_folder))
     lbl = [LABELS.index(l_folder)] * len(pth)
     label.extend(lbl)
  return image_path, images, label
TRAIN_PATH = '/content/gdrive/MyDrive/Test_data/Test_data/Train/'
VAL_PATH = '/content/gdrive/MyDrive/Test_data/Test_data/Val/'
TEST_PATH = '/content/gdrive/MyDrive/Test_data/Test_data/Test/'
train_img_path, train_images, train_lbl = read_imgfilepath_label(TRAIN_PATH)
val_img_path, val_images, val_lbl = read_imgfilepath_label(VAL_PATH)
test_img_path = glob.glob(os.path.join(TEST_PATH, '*.jpg')) 
NUM_OF_LABELS = len(np.unique(train_lbl)) 

def plot(xvalue, yvalue, xlabl, ylabl,color):
  plt.plot(xvalue, yvalue, color=color)
  plt.xlabel(xlabl)
  plt.ylabel(ylabl)
  
def data_augmentation(image, p=0.5):
  #p -> prbability of performing a given augmnetation
  if p>= 0 and p <= 0.5:
    contrast=iaa.GammaContrast(gamma=2.0)
    image =contrast.augment_image(image)
  if p >= 0.2 and p <= 0.6:
    height, width = image.shape[:2]
    h_center, w_center = height//2, width//2
    image = image[(h_center-35):(h_center+35),(w_center-35):(w_center+35),:]   
    image = cv2.resize(image, (96, 96), interpolation = cv2.INTER_AREA)
    #print(image.shape)
  if p >=0.3 and p<=0.8:
    flip_hr=iaa.Fliplr(p=1.0)
    image= flip_hr.augment_image(image)
  if p >=0.5 and p<=0.7:
    flip_vr=iaa.Flipud(p=1.0)
    image= flip_vr.augment_image(image)
  #print(image.shape,'image kis hapoee')
  return image
  
class dataloader(torch.utils.data.Dataset):
  def __init__(self, list_IDs, labels, train_or_vald):
    self.labels = labels
    self.list_IDs =list_IDs
    self.train_or_vald = train_or_vald

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    #print(self.list_IDs)
    ID = self.list_IDs[index]
    label = self.labels[index]
    image = cv2.imread(ID)
    ###Scale image
    image = image/255
    #visualize(image)
    #image = np.transpose(image, (2,0,1))
    #print(image.shape, 'image shape')
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    transform = transforms.ToTensor()
    if self.train_or_vald == 'train':
      p = np.random.uniform(low = 0, high = 1)
      x1 = image[:,:,0]
      #x1 = data_augmentation(image, p)[:,:,0]
      #####visualisation after augmentation##############
      #visualize(np.transpose(x1,(1,2,0)))
      #x1 = torch.from_numpy(x1.copy())
      #x1 = torch.as_tensor(x1, dtype = torch.float32)
      x1 = transform(x1.copy())
      #y = torch.zeros(NUM_OF_LABELS)
      #y[label] = 1
      y = label

    elif self.train_or_vald == 'vald':
      image = image[:,:,0]
      x1 = transform(image.copy())
      y = label 
    return x1, y


###############model##############
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 22 * 22, 5)
        self.fc2 = nn.Linear(96, 5)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.batch_norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        #print(x.shape, 'x ki shape')
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape, 'x ki shape')
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x
  train_data = dataloader(train_img_path, train_lbl,'train')
batch_size = 900
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle= True)

val_data = dataloader(val_img_path, val_lbl,'vald')
val_data_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size = 500, shuffle= True)
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
globaliter = 0
net = Net().to(device=device)
criterion = nn.CrossEntropyLoss()
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []
optimizer = optim.Adam(net.parameters(), betas=(0.90, 0.98), lr=0.00001)
epochs=100
for epoch in range(epochs):
  net.train()
  correct = 0
  total = 0
  count = 0
  #with tqdm(total=NUM_OF_LABELS, desc=f'Epoch{epoch +1}/{epochs}', unit='img') as pbar:
  for i, (input, labl) in enumerate(train_data_loader):
    running_loss = 0.0
    globaliter += 1
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    #print(input.shape)
    input = input.to(device=device, dtype=torch.float32)
    outputs = net(input)
    #print(outputs)
    #print(labl)
    #pbar.update(input.shape[0])
    labl = labl.to(device=device)
    loss = criterion(outputs, labl)
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss += loss.item()
    total += labl.size(0)
    #print(torch.max(outputs, 1))
    _, predicted = torch.max(outputs, 1)
    #print(predicted, labl)
    correct += (predicted == labl).sum().item()
    count +=1
    #pbar.set_postfix(**{'loss (batch)': loss.item()})
  train_loss = running_loss/count
  train_acc = (100 * correct / total)
  training_loss.append(train_loss)
  training_accuracy.append(train_acc)
  print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
        #running_loss = 0.0
  correct = 0
  total = 0
  net.eval()
  vald_loss = 0
  count = 0
  for data in val_data_loader:
    print('yes')
    images, labels = data
    # calculate outputs by running images through the network
    images = images.to(device=device, dtype=torch.float32)
    outputs = net(images)
    labels = labels.to(device=device)
    loss = criterion(outputs, labels.to(device))
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    vald_loss += loss.item()
    count+=1
  vald_loss = vald_loss/count
  vald_acc = (100 * correct / total)
  validation_loss.append(vald_loss)
  validation_accuracy.append(vald_acc)
  print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
  writer.add_scalar('Train Loss', train_loss, int(epoch))
  writer.add_scalar('Valid Loss', vald_loss, int(epoch))
  writer.add_scalar('Train Accuracy', train_acc, int(epoch))
  writer.add_scalar('Valid Accuracy', vald_acc, int(epoch))
  

epchs = [x for x in range(epoch+1)]
plot(epchs, training_accuracy, 'Accuracy', 'Epoch','g')
plt.figure()
plot(epchs, training_loss, 'Loss', 'Epoch','g')
plt.figure()
plot(epchs, validation_accuracy, 'Accuracy', 'Epoch','g')
plt.figure()
plot(epchs, validation_loss, 'Loss', 'Epoch','g')
