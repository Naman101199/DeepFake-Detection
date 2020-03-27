import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
import math
from torchvision import transforms
import torchvision.models as models
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn

model = models.resnext50_32x4d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)
count = 0
for name, child in model.named_children():
  count+=1
  for param in child.parameters():
    if(count<7):
      param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_function = nn.BCEWithLogitsLoss()
BATCH_SIZE = 64
EPOCHS = 20

#Converting to tensors

features = torch.as_tensor(features).view(-1,3,224,224)
labels = torch.from_numpy(labels).float().to(device)
labels = labels.reshape(-1,1)

#Train-test split

Features_train = features[0:int(np.ceil(len(features)*0.8))]
Features_val = features[int(np.ceil(len(features)*0.8)):len(features)]
Labels_train = labels[0:int(np.ceil(len(labels)*0.8))]
Labels_val = labels[int(np.ceil(len(labels)*0.8)):len(labels)]

#Training

for epoch in tqdm(range(EPOCHS)):
  for i in range(0, len(Features_train), BATCH_SIZE):
    batch_X = Features_train[i:i+BATCH_SIZE].view(-1,3,224,224)
    batch_y = Labels_train[i:i+BATCH_SIZE]
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.001*(1/(1+epoch)))
    optimizer.zero_grad()
    outputs = model(batch_X.float())
    loss = loss_function(outputs, batch_y)
    loss.backward()
    optimizer.step()
  print(loss)
  
torch.save(model,"Resnext.pth")
