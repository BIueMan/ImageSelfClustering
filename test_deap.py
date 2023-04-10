import cv2
from main import *
from funcrions.semi_supervised_DL import *
from funcrions.semi_supervised_learning import *
import torch
import torch.nn as nn
import torch.optim as optim

## test main
if __name__ == "__main__":
    # Initialize model and optimizer
    img_test = cv2.imread('images/Amanita_muscaria_test.jpeg')
    print(img_test.shape)

    splited = split_image(img_test, [200,200], [100, 100])
    splited = torch.tensor(splited)
    model = CNN(splited[0, 0, :, :, :])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example usage
    output = model(splited)
    loss = model.loss_func(output)
    loss.backward()
    optimizer.step()