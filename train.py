import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import PIL

from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from nyu_dataloader import LoadData, NYUDataset
from model import CoarseNetwork, FineNetwork
from metrics import Metrics_Calculate
from loss import Train_Loss

%matplotlib inline
%load_ext autoreload
%autoreload 2


param =0.5 # 0 1
# # learning_rate = {'coarse': [0.001,0.001,0.001,0.001,0.001,0.1,0.1], 'fine':[0.01,0.1,0.01]}
learning_rate = 0.001
num_epochs = {'coarse':30, 'fine':30}

criterion = Train_Loss(param)

optimizer = {'coarse': torch.optim.SGD(net['coarse'].parameters(), lr = 0.001, momentum=0.9, weight_decay= 5e-4),
             'fine':torch.optim.SGD(net['fine'].parameters(), lr = 0.001, momentum=0.9, weight_decay= 5e-4)}

# optimizer = {'coarse': torch.optim.Adam(net['coarse'].parameters(), lr = 0.001),
#              'fine':torch.optim.Adam(net['fine'].parameters(), lr = 0.001)}

optimizer = {}
optimizer['coarse'] = torch.optim.SGD([
                                       {'params': net['coarse'].coarse1.parameters(), 'lr': 0.001}, {'params': net['coarse'].coarse2.parameters(), 'lr': 0.001},
                                       {'params': net['coarse'].coarse3.parameters(), 'lr': 0.001}, {'params': net['coarse'].coarse4.parameters(), 'lr': 0.001},
                                       {'params': net['coarse'].coarse5.parameters(), 'lr': 0.001}, {'params': net['coarse'].coarse6.parameters(), 'lr': 0.1},
                                       {'params': net['coarse'].coarse7.parameters(), 'lr': 0.1}
                                       ], lr=0.001, momentum=0.9, weight_decay= 5e-4)

optimizer['fine'] = torch.optim.SGD([
                                     {'params': net['fine'].fine1.parameters(), 'lr': 0.001},
                                     {'params': net['fine'].fine2.parameters(), 'lr': 0.01},
                                     {'params': net['fine'].fine3.parameters(), 'lr': 0.001}
                                     ], lr=0.001, momentum=0.9, weight_decay= 5e-4)

# optimizer = {}
# optimizer['coarse'] = torch.optim.Adam([
#                                        {'params': net['coarse'].coarse1.parameters(), 'lr': 0.001}, {'params': net['coarse'].coarse2.parameters(), 'lr': 0.001},
#                                        {'params': net['coarse'].coarse3.parameters(), 'lr': 0.001}, {'params': net['coarse'].coarse4.parameters(), 'lr': 0.001},
#                                        {'params': net['coarse'].coarse5.parameters(), 'lr': 0.001}, {'params': net['coarse'].coarse6.parameters(), 'lr': 0.1},
#                                        {'params': net['coarse'].coarse7.parameters(), 'lr': 0.1}
#                                        ], lr=0.001)

# optimizer['fine'] = torch.optim.Adam([
#                                      {'params': net['fine'].fine1.parameters(), 'lr': 0.001},
#                                      {'params': net['fine'].fine2.parameters(), 'lr': 0.01},
#                                      {'params': net['fine'].fine3.parameters(), 'lr': 0.001}
#                                      ], lr=0.001)

def train(mode, net, optimizer):

    best_test_loss = float("inf")
    val_loss_list = []
    train_loss_list = []
    # Loop over the dataset for multiple epochs
    for epoch in range(num_epochs[mode]):
        net[mode].train()
        running_loss = 0.0
        start_time = time.time()

        print('\nStarting epoch %d / %d' % (epoch + 1, num_epochs[mode]))

        # For each mini-batch...
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer[mode].zero_grad()
            if mode == 'coarse':
                outputs  = net['coarse'](inputs)
            elif mode == 'fine':
                with torch.no_grad():
                    net['coarse'].eval()
                    coarse_outputs = net['coarse'](inputs)
                outputs = net['fine'](inputs, coarse_outputs.detach())

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer[mode].step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        train_loss_list.append(running_loss)

        # save model every 5 epochs
        # if epoch %5 == 4 and load_network_path is not None:
        #   torch.save(net[mode].state_dict(), load_network_path[mode])

        # evaluate the network on the validation dataset
        with torch.no_grad():
            val_loss = 0.0
            net[mode].eval()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if mode == 'coarse':
                    outputs  = net['coarse'](inputs)
                elif mode == 'fine':
                    net['coarse'].eval()
                    coarse_outputs = net['coarse'](inputs)
                    outputs = net['fine'](inputs, coarse_outputs)
                loss = criterion(outputs,labels)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)
            # Metrics: t1, t2, t3, abs_error, squared_error, rmse_linear, rmse_log
            t1, t2, t3, abs_error, squared_error, rmse_linear, rmse_log =  Metrics_Calculate(outputs, labels)
            print("epoch:", epoch + 1, ", training loss:", running_loss, "validation loss:", val_loss)
            if epoch % 10 == 9:
                print("\n------------Validation--------------")
                print("Threshold < 1.25:", t1)
                print("Threshold < 1.25^2:", t2)
                print("Threshold < 1.25^3:", t3)
                print("abs_relative_difference:", abs_error.item())
                print("squared_relative_difference:", squared_error.item())
                print("RMSE (linear):", rmse_linear.item())
                print("RMSE (log):", rmse_log.item())
                print("RMSE (log, scale inv.):", val_loss)
                print("---------------------------------------")

        # training_time = time.time() - start_time
        # print("Training time: %d min %d s"% (training_time//60, training_time % 60))

    return net, train_loss_list, val_loss_list

# Train the coarse network
net, train_losses, val_losses= train('coarse', net, optimizer)
plot_loss(train_losses, val_losses)

# Train the fine network
net, train_losses, val_losses= train('fine', net, optimizer)
plot_loss(train_losses, val_losses)

# Test the model and output samples
test_loss = 0.0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    net["coarse"].eval()
    net["fine"].eval()

    with torch.no_grad():
        coarse_outputs = net['coarse'](inputs)
        fine_outputs = net['fine'](inputs, coarse_outputs)

    loss = criterion(fine_outputs, labels)
    test_loss += loss.item()

test_loss /= len(test_loader)
print("Test loss: ", test_loss)
t1, t2, t3, abs_error, squared_error, rmse_linear, rmse_log =  Metrics_Calculate(fine_outputs, labels)
print("\n------------Validation--------------")
print("Threshold < 1.25:", t1)
print("Threshold < 1.25^2:", t2)
print("Threshold < 1.25^3:", t3)
print("abs_relative_difference:", abs_error.item())
print("squared_relative_difference:", squared_error.item())
print("RMSE (linear):", rmse_linear.item())
print("RMSE (log):", rmse_log.item())
print("RMSE (log, scale inv.):", test_loss)
print("---------------------------------------")
