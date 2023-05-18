#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script provides an example for timing a training loop.

Usage: -

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2023-05-17
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import argparse
import sys
import os
import time


# @profile
def main(args):

    # Print the arguments.
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers:    {args.num_workers}")

    # Set the random seed for PyTorch and NumPy.
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(args.num_workers)
    np.random.seed(0)

    # Check torch CUDA
    print(f"\n{'torch.cuda.is_available():'.ljust(32)}"
      f"{torch.cuda.is_available()}")
    print(f"{'torch.cuda.device_count():'.ljust(32)}"
        f"{torch.cuda.device_count()}")
    print(f"{'torch.cuda.current_device():'.ljust(32)}"
        f"{torch.cuda.current_device()}")
    print(f"{'torch.cuda.device(0):'.ljust(32)}"
        f"{torch.cuda.device(0)}")
    print(f"{'torch.cuda.get_device_name(0):'.ljust(32)}"
        f"{torch.cuda.get_device_name(0)}")
    print(f"{'torch.backends.cudnn.benchmark:'.ljust(32)}"
        f"{torch.backends.cudnn.benchmark}")

    # Check CPUs available (for num_workers).
    print(f"\n{'os.sched_getaffinity:'.ljust(32)}"
          f"{len(os.sched_getaffinity(0))}")
    print(f"{'os.cpu_count():'.ljust(32)}"
          f"{os.cpu_count()}\n")

    # Set device (GPU if available, else CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformations for training and testing datasets.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset and apply transformations.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Display the size of the training dataset.
    print(f"\nTraining dataset size: {len(trainset)}")
    print(f"Number of batches in the training dataset: {len(trainloader)}")

    # Load the ResNet-18 model.
    model = resnet18(weights=None)
    num_features = model.fc.in_features     # Get the number of features in the last convolutional layer.
    model.fc = nn.Linear(num_features, 10)  # Modify the last fully connected layer for 10 classes.
    model = model.to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop.
    start_time = time.time()
    for epoch in range(args.epochs):  # Number of epochs.
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # Print average loss every 200 mini-batches.
                print('[%2d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    end_time = time.time()
    training_time = end_time - start_time
    print("Training finished! Total time: %.2f seconds" % training_time)

    # Evaluation loop.
    correct = 0
    total = 0
    print(f"\nTesting dataset size:  {len(testset)}")
    print(f"Number of batches in the testing dataset:  {len(testloader)}")
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))

    return 0


if __name__ == "__main__":

    # Get arguments.
    parser = argparse.ArgumentParser(
        description="Script for testing PyTorch scalability."
    )

    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='number of epochs for training (default: 10).')

    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='number of images in a batch during training '
                            '(default: 64).')

    parser.add_argument('--num_workers', '-w', type=int, default=4,
                        help='number of subprocesses to use for data loading. '
                        '0 means that the data will be loaded in the '
                        'main process (default: 4).')

    args = parser.parse_args(sys.argv[1:])

    # Main function.
    sys.exit(main(args))
