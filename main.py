import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch.serialization


torch.manual_seed(21394)

target_size = (128, 128)
target_format = 'RGB'

if target_format == 'L':
    n_channels = 1
elif target_format == 'RGB':
    n_channels = 3

#----------- Data processing
def load_and_preprocess(image_path):
    image = Image.open(image_path).convert(target_format)
    image = image.resize(target_size)
    image = torch.tensor(np.array(image)/255.0, dtype=torch.float32)
    return image

def load_data(data_dir):
    data = []
    labels = []
    classes = sorted(os.listdir(data_dir))
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image = load_and_preprocess(image_path)
                data.append(image)
                labels.append(idx)

    labels = torch.tensor(labels)
    data = torch.tensor(np.array(data))
    data = torch.permute(data, (0, 3, 1, 2)).contiguous()
    return data, labels, classes

data, labels, classes = load_data('data/train')

batch_size = 6

def get_batch():
    ix = torch.randint(0, data.shape[0], (batch_size, ))
    return data[ix], labels[ix]

# For calculating the loss on some data that the model never seen
@torch.no_grad()
def loss_eval(data_path):
    model.eval()
    data, labels, classes = load_data(data_path)
    logits, loss = model(data, labels)
    model.train()
    return loss

n_classes = len(classes)
n_hidden = 128

#---------- NN construction

class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(n_channels, 16, 3, stride=1),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(16, 4, 4, stride=1),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU())

        self.projection = nn.Conv2d(n_channels, 4, kernel_size=12, stride=4, bias=False)

        self.fc = nn.Sequential(nn.Linear(4*30*30, n_hidden, bias=False),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden, bias=True),
                                nn.Tanh(),
                                nn.Linear(n_hidden, n_hidden//2, bias=True),
                                nn.Dropout(p=0.2),
                                nn.Tanh(),
                                nn.Linear(n_hidden//2, n_classes, bias=True))

    def forward(self, train_data, y=None):
        iden1 = train_data
        iden1 = self.projection(iden1)
        x = self.conv(train_data)
        x = x + iden1
        A, B, C, D = x.shape
        x = x.view(A, -1)
        logits = self.fc(x)
        logits = logits
        if y is not None:
            loss = F.cross_entropy(logits, y)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def classify(self, img_path):
        self.eval()
        img = load_and_preprocess(img_path)
        x = torch.stack((img, ), dim=0)
        x = torch.permute(x, (0, 3, 1, 2)).contiguous()
        logits, loss = self(x)
        probs = F.softmax(logits, dim=1)
        ix = torch.argmax(probs, dim=1)
        self.train()
        return classes[ix]

#---------- Initialization

model = Classification()

# --------- Result

torch.serialization.add_safe_globals([Classification])
torch.load("model.pth")
result = model.classify('data/test/humanpic.jpg')

print(result)
