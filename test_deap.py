import cv2
from main import *
from funcrions.semi_supervised_DL import *
from funcrions.semi_supervised_learning import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import itertools

# Check PyTorch
print(f"Using PyTorch version {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

## test main
if __name__ == "__main__":
    # Initialize model and optimizer
    img_test = cv2.imread('images/Amanita_muscaria_test.jpeg')
    print(img_test.shape)

    splited = split_image(img_test, [200,200], [200, 200])
    test_image =  torch.tensor(splited)[0, 0, :, :, :].permute(2, 0, 1).unsqueeze(0).float()
    model = CNN(test_image) # init model for testing
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example usage
    output = model(test_image)

    # Get Loss Func
    x_loc = [0, 0]
    sigma = simga_dist(nearest_neighbors(splited), splited.shape)
    label_matrix = [[0, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 3, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
    S = np.array(label_matrix)

    # estimate phi_i
    M, N, _, _, _ = splited.shape
    num_of_labels = np.max(label_matrix)
    phi_output_saved = np.zeros([M, N, num_of_labels])
    for label_num in range(num_of_labels):
        # model = CNN(test_image)
        # train model
        for repet in tqdm(range(2)):
            sum_lost = torch.tensor([[0]], dtype= torch.float)
            for x_loc in tqdm(itertools.product(range(M), range(N))):
                optimizer.zero_grad()
                loss = model.loss_func(x_loc, splited, sigma, S, 1)
                sum_lost += loss
                loss.backward()
                optimizer.step()
            print(sum_lost)

        print('extract phi output: %d', label_num)
        for x_loc in tqdm(itertools.product(range(M), range(N))):
            image = torch.tensor(splited[*x_loc, :, :, :]).permute(2, 0, 1).unsqueeze(0).float()
            phi_output_saved[*x_loc, label_num] = model.forward(image)
    
    # use phi_i in order to extract Label
    Label = np.argmax(phi_output_saved, axis=2)
    print(Label)
    print(phi_output_saved)
