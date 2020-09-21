import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import argparse
import time

from model import RNNClassifier, linearRegression
from util import csv_to_data, load_df


def get_device():
    """
    gpu 사용 가능하면 gpu device를, 아니면 cpu device를 출력
    :return: usable device, gpu vs cpu
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Fix Seed
    start = time.time()
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)

    # Preprocessing
    slicer = 9000
    train_data, test_data = csv_to_data(slicer)
    train_data, test_data = load_df(slicer)

    batch_size = 16
    device = get_device()

    train_input_list = train_data['input'].to_list()
    test_input_list = test_data['input'].to_list()
    print("# of Train data : ", len(train_input_list), ", # of Test data : ", len(test_input_list))

    # Make Tensor and Data Loader
    train_input_ = torch.Tensor(train_input_list).to(device=device)
    train_target_ = torch.Tensor(np.array(train_data['label'])).long().to(device=device)
    train = torch.utils.data.TensorDataset(train_input_, train_target_)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_input_ = torch.Tensor(test_input_list).to(device=device)
    test_target_ = torch.Tensor(np.array(test_data['label'])).long().to(device=device)
    test = torch.utils.data.TensorDataset(test_input_, test_target_)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    n_dim = 2
    n_hidden = 4096
    n_layer = 1
    n_label = 2
    learning_rate = 0.0001

    ## Experimental Setting :: Model, Optimizer, Loss function
    # model = RNNClassifier(device, n_dim, n_hidden, n_layer, n_label)
    model = linearRegression(slicer, n_dim, n_hidden, n_layer, n_label)
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 5000
    test_cycle = 5
    train_loss_list = []
    test_loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0
    for epoch in range(num_epochs):
        # print("Iteration {} : time : {}".format(epoch+1, time.time()-start))
        # Training
        train_correct = 0
        train_total = 0
        loss_list = list()
        for i, (train_inputs, train_labels) in enumerate(train_loader):
            train_tensor = train_inputs.view(train_labels.shape[0], -1, n_dim)

            optimizer.zero_grad()
            train_outputs = model(train_tensor)
            train_loss = criterion(train_outputs, train_labels)
            loss_list.append(train_loss.data)

            train_predicted = torch.max(train_outputs.data, 1)[1]
            train_total += train_labels.size(0)
            train_correct += (train_predicted == train_labels).sum()

            train_loss.backward()
            optimizer.step()
        accuracy = 100 * train_correct / float(train_total)

        train_loss = sum(loss_list) / float(len(train_loader))
        train_loss_list.append(train_loss)
        count += 1
        print('Iteration: {}  Train Loss: {}  Accuracy: {} %'.format(count, train_loss, accuracy))

        # Validation / Test
        if count % test_cycle == 0:
            test_correct = 0
            test_total = 0
            loss_list2 = list()
            for test_inputs, test_labels in test_loader:
                test_tensor = test_inputs.view(test_labels.shape[0], -1, n_dim)
                test_outputs = model(test_tensor)
                test_loss = criterion(test_outputs, test_labels)
                loss_list2.append(test_loss)

                test_predicted = torch.max(test_outputs.data, 1)[1]
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum()
            accuracy = 100 * test_correct / float(test_total)

            test_loss = sum(loss_list2) / float(len(test_loader))
            test_loss_list.append(test_loss)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            print('Iteration: {}  Valid Loss: {}  Accuracy: {} %'.format(count, test_loss, accuracy))
