# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        #layer 1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #layer 2
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #layer 3
        self.flatten_1 = nn.Flatten()

        #layer 4
        self.linear_1 = nn.Linear(in_features=16*5*5, out_features=256)
        self.relu_3 = nn.ReLU()

        #layer 5
        self.linear_2 = nn.Linear(in_features=256, out_features=128)
        self.relu_4 = nn.ReLU()
        
        #layer 6
        self.linear_3 = nn.Linear(in_features=128, out_features=100)

    def forward(self, x):
        # certain operations
        #stage 1
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)
        stage1_shape = list(x.shape)

        #stage 2
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)
        stage2_shape = list(x.shape)

        #stage 3
        x = self.flatten_1(x)
        stage3_shape = list(x.shape)

        #stage 4
        x = self.linear_1(x)
        x = self.relu_3(x)
        stage4_shape = list(x.shape)

        #stage 5
        x = self.linear_2(x)
        x = self.relu_4(x)
        stage5_shape = list(x.shape)

        # Stage 6
        x = self.linear_3(x)
        stage6_shape = list(x.shape)

        #Add to dict
        shape_dict = {
            1: stage1_shape,
            2: stage2_shape,
            3: stage3_shape,
            4: stage4_shape,
            5: stage5_shape,
            6: stage6_shape
        }

        return x, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for name, p in model.named_parameters():
        i = 1
        for e in list(p.size()):
            i *= e
        model_params += i
    return model_params/1000000


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
