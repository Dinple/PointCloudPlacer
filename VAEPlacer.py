import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F

from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook as tqdm

from canvas import *
from box import *
from net import *


class VAEPlacer:
    """
    VAEPlacer is a utility class that is used to place PCB components on the canvas.
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
        # for each feature, normalize the data to be between 0 and 1 avoiding division by 0
        x = (x-np.min(x))/(np.max(x)-np.min(x))

        y = box_dataset[:, -1]
        # convert to tensor
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        # create the model
        feature_dim = x.size()[1]
        compression_dim = 24
        self.model = VAE(
            FC_Encoder(img_size=feature_dim, output_size=compression_dim),
            FC_Decoder(img_size=feature_dim, input_size=compression_dim),
        )
        # create the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # loss function
        def loss_fn(recon_x, x, mu, logvar, beta=1., verbose=False):
            # print("     [loss info] recon_x.size() ", recon_x.size(), " x.size() ", x.size()) 
            assert recon_x.size() == x.size()
            # bcefn = nn.BCELoss()
            # print("     [loss info] recon_x ", recon_x, " x ", x)
            BCE = F.binary_cross_entropy(recon_x, x, reduction="sum") # the output will be summed
            # BCE = F.binary_cross_entropy(recon_x, x, reduction="mean") # the output will be averaged
            KLD = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).sum()
            # make sure loss is positive
            assert BCE.item() >= 0
            assert KLD.item() >= 0

            print("     [loss info] BCE ", BCE.item(), " KLD ", KLD.item(), " beta KLD", beta * KLD.item()) if verbose else None
            return BCE + beta * KLD
        
        # train the model
        for epoch in range(10000):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            recon_x, mu, logvar = self.model(x)
            # print("     [train info] recon_x.size() ", recon_x.size(), " x.size() ", x.size())
            loss = loss_fn(recon_x, x, mu, logvar, beta=1., verbose=False)
            loss.backward()
            optimizer.step()
            # print the loss
            if epoch % 100 == 0:
                print("Epoch %d: %f" % (epoch, loss.item()))

        # plot the latent space by using the encoder
        # get the latent space
        latent_space = []
        for i in range(len(x)):
            # get the latent space
            mu, logvar = self.model.encoder(x[i])
            z = self.model.reparameterise(mu, logvar)
            latent_space.append(z.detach().numpy())
        latent_space = np.array(latent_space).squeeze()
        print("latent_space ", latent_space.shape)
        print("y ", y)
        # plot the latent space using tsne
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(latent_space)
        print("tsne_results ", tsne_results.shape)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y.detach().numpy(), cmap=plt.cm.get_cmap('jet', 10))
        plt.show()
    
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


class FC_Encoder(nn.Module):
    def __init__(self, img_size=784, output_size=2):
        super(FC_Encoder, self).__init__()
        self.img_size = img_size
        self.output_size = output_size
        self.nn = nn.Sequential(
            nn.Linear(self.img_size, self.output_size),
        )
        self.mu_layer = nn.Linear(
            self.output_size, self.output_size
        )
        self.logvar_layer = nn.Linear(
            self.output_size, self.output_size
        )

    def forward(self, x):
        # make sure to flatten the image first
        x = x.view(-1, self.img_size)
        # print("     [FC_Encoder info] x.size() ", x.size())
        x = self.nn(x)
        # print("     [FC_Encoder info] x ", x)
        mu = self.mu_layer(x)
        # print("     [FC_Encoder info] mu ", mu)
        logvar = self.logvar_layer(x)
        # print("     [FC_Encoder info] logvar ", logvar)
        return mu, logvar

class FC_Decoder(nn.Module):
    def __init__(self, img_size=784, input_size=2):
        super(FC_Decoder, self).__init__()
        self.img_size = img_size
        self.input_size = input_size
        self.nn = nn.Sequential(
            nn.Linear(input_size, self.img_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # reshape to square image
        recon = self.nn(z)
        # print("     [FC_Decoder info] recon ", recon)
        # print("     [FC_Decoder info] recon.size() ", recon.size())
        # recon = recon.view(-1, 1, int(np.sqrt(self.img_size)), int(np.sqrt(self.img_size)))
        return recon
    
class VAE(nn.Module):
    def __init__(self, enc, dec):
        super(VAE, self).__init__()
        self.encoder = enc
        self.decoder = dec

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            # mean + std * eps
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # This function should be modified for the DAE and VAE
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar