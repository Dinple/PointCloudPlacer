import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F

from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm_notebook as tqdm

from canvas import *
from box import *
from net import *


class NeuroPlacer:
    """
    NeuroPlacer is a utility class that is used to place PCB components on the canvas.
    It is inspired by training neural networks to generate latent representations of the PCB components.
    In this framework, the PCB components are the input, and the latent representations are the output.
    Each net is a class, and each box is an instance of the class.
    The placement of the boxes is the latent representation of the net.
    The placement of the boxes is updated by backpropagation.
    """

    def __init__(self, canvas: PCBCanvas):
        self.canvas = canvas

    def get_box_dataset(self):
        """
        prepare the dataset for the boxes
        """
        # feature 1: box width
        box_width = []
        for box in self.canvas.iter_box():
            box_width.append(box.width)
        box_width = np.array(box_width)
        # feature 2: box height
        box_height = []
        for box in self.canvas.iter_box():
            box_height.append(box.height)
        box_height = np.array(box_height)
        # feature 3: box cx
        box_cx = []
        for box in self.canvas.iter_box():
            box_cx.append(box.cx)
        box_cx = np.array(box_cx)
        # feature 4: box cy
        box_cy = []
        for box in self.canvas.iter_box():
            box_cy.append(box.cy)
        box_cy = np.array(box_cy)
        # feature 5: box rotation
        box_rotation = []
        for box in self.canvas.iter_box():
            box_rotation.append(box.rotation)
        box_rotation = np.array(box_rotation)
        # feature 6: box net name => change to categorical
        box_net_name = []
        box_net_name_dict = {}
        for box in self.canvas.iter_box():
            # assign a number to each net name
            if box.net_name not in box_net_name_dict:
                box_net_name_dict[box.net_name] = len(box_net_name_dict)
            box_net_name.append(box_net_name_dict[box.net_name])
        box_net_name = np.array(box_net_name)

        # put the features into a matrix
        box_dataset = np.vstack(
            (box_width, box_height, box_cx, box_cy, box_rotation, box_net_name)
        ).T
        return box_dataset

    def get_net_dataset(self):
        """
        prepare the dataset for the nets
        """
        # feature 1: net source box width
        net_source_box_width = []
        for net in self.canvas.iter_net():
            net_source_box_width.append(self.canvas.get_box(net.source_box_name).width)
        net_source_box_width = np.array(net_source_box_width)
        # feature 2: net source box height
        net_source_box_height = []
        for net in self.canvas.iter_net():
            net_source_box_height.append(
                self.canvas.get_box(net.source_box_name).height
            )
        net_source_box_height = np.array(net_source_box_height)
        # feature 3: net source box cx
        net_source_box_cx = []
        for net in self.canvas.iter_net():
            net_source_box_cx.append(self.canvas.get_box(net.source_box_name).cx)
        net_source_box_cx = np.array(net_source_box_cx)
        # feature 4: net source box cy
        net_source_box_cy = []
        for net in self.canvas.iter_net():
            net_source_box_cy.append(self.canvas.get_box(net.source_box_name).cy)
        net_source_box_cy = np.array(net_source_box_cy)
        # feature 5: net source box rotation
        net_source_box_rotation = []
        for net in self.canvas.iter_net():
            net_source_box_rotation.append(
                self.canvas.get_box(net.source_box_name).rotation
            )
        net_source_box_rotation = np.array(net_source_box_rotation)
        # feature 6: net source box net name
        net_source_box_net_name = []
        for net in self.canvas.iter_net():
            net_source_box_net_name.append(
                self.canvas.get_box(net.source_box_name).net_name
            )
        net_source_box_net_name = np.array(net_source_box_net_name)

        # feature 7: net sink box width
        net_sink_box_width = []
        for net in self.canvas.iter_net():
            net_sink_box_width.append(
                self.canvas.get_box(net.get_sink_box_name(1)).width
            )
        net_sink_box_width = np.array(net_sink_box_width)
        # feature 8: net sink box height
        net_sink_box_height = []
        for net in self.canvas.iter_net():
            net_sink_box_height.append(
                self.canvas.get_box(net.get_sink_box_name(1)).height
            )
        net_sink_box_height = np.array(net_sink_box_height)
        # feature 9: net sink box cx

    def train(self):
        """
        train the model
        """
        # get the box dataset
        box_dataset = self.get_box_dataset()
        # train the model
        self.train_model(box_dataset)

    def train_model(self, box_dataset):
        """
        train the model
        """
        # last column is the y value
        x = box_dataset[:, :-1]
        y = box_dataset[:, -1]
        # convert to tensor
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        # create the model
        self.model = BoxModel(
            num_box_features=len(x[0]), num_box_latent=100, num_box_output=1
        )
        # create the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # create the loss function : use triplet loss
        # loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        loss_fn = nn.MSELoss()
        # train the model
        for epoch in tqdm(range(10000)):
            # zero the gradient
            optimizer.zero_grad()
            # forward pass
            y_pred = self.model(x)
            # compute the loss
            loss = loss_fn(y_pred, y)
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()
            # print the loss
            if epoch % 100 == 0:
                print("Epoch %d: %f" % (epoch, loss.item()))
        # get the latent representation
        latent = self.model.get_latent(x)
        # get the output
        output = self.model.get_output(latent)
        # plot the latent representation
        plt.scatter(
            latent[:, 0].detach().numpy(),
            latent[:, 1].detach().numpy(),
            c=y.detach().numpy(),
        )

        return latent, output
    
    def evaluate(self):
        """
        evaluate the model
        """
        # get the box dataset
        box_dataset = self.get_box_dataset()
        # evaluate the model
        self.evaluate_model(box_dataset)

    def evaluate_model(self, box_dataset):
        '''
        evaluate the model
        '''
        # last column is the y value
        x = box_dataset[:, :-1]
        y = box_dataset[:, -1]
        # convert to tensor
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        # compare the output with the y value
        y_hat = self.model(x).flatten()
        accuracy = torch.sum(torch.abs(y_hat - y) < 0.5) / len(y)
        print(f"Accuracy: {torch.sum(torch.abs(y_hat - y) < 0.5)} / {len(y)} = {accuracy}")





class BoxModel(nn.Module):
    """
    BoxModel is a neural network model that is used to generate the latent representation of the boxes.
    """

    def __init__(self, num_box_features=6, num_box_latent=2, num_box_output=2):
        super(BoxModel, self).__init__()
        # the input layer
        self.input_layer = nn.Linear(num_box_features, num_box_latent)
        # the output layer
        self.output_layer = nn.Linear(num_box_latent, num_box_output)

    def forward(self, x):
        """
        forward pass
        """
        # input layer
        x = self.input_layer(x)
        # activation function
        x = F.relu(x)
        # output layer
        x = self.output_layer(x)
        return x

    def get_latent(self, x):
        """
        get the latent representation
        """
        # input layer
        x = self.input_layer(x)
        # activation function
        x = F.relu(x)
        return x

    def get_output(self, x):
        """
        get the output
        """
        # output layer
        x = self.output_layer(x)
        return x
