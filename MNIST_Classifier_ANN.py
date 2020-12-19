# PyTorch to Classify MNIST digits

# importing packages
import numpy as np
import torch, torchvision
from visdom import Visdom  # vizualizing losses in Model
from torch import nn, optim
from torch.autograd import Variable  # convert images to variable

# data loader
T = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)  # creating a transformation T to convert images to tensor format
mnist_data = torchvision.datasets.MNIST(
    "mnist_data", transform=T, download=True
)  # downloading data

m_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=128)


# neural network
class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.fin_linear = nn.Linear(50, 10)

        self.relu = nn.ReLU()

    def forward(
        self, images
    ):  # defining the forward pass, Pytorch constructs the backward itself
        x = images.view(-1, 28 * 28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.fin_linear(x))
        return


# training loop
model = Mnet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params=params, lr=0.001)

n_epochs = 3
n_iterations = 0

vis = Visdom()
vis_window = vis.line(np.array([0]), np.array([0]))
for e in range(n_epochs):
    for i, (images, labels) in enumerate(m_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)

        model.zero_grad()
        loss = cec_loss(output, labels)
        loss.backward()  # compute gradients to optimizer to update using learning rate

        optimizer.step()
        n_iterations += 1

        vis.line(
            np.array([loss.item()]),
            np.array([n_iterations]),
            win=vis_window,
            update="append",
        )
        break