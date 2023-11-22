import os
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

class PCBPlacer:
    def __init__(self, canvas:PCBCanvas):
        self.canvas = canvas

    # initialization methods
    def init_plc_center(self):
        '''
        init the boxes to the center of the canvas 
        '''
        # init boxes
        for box in self.canvas.iter_box():
            box.set_center_xy(self.canvas.width / 2, self.canvas.height / 2)

    def init_plc_spectral(self):
        '''
        init the boxes to the spectral embedding of the canvas
        '''
        # init boxes
        cxy = self.canvas.get_cxy()
        max_cxy = np.max(cxy, axis=0)
        min_cxy = np.min(cxy, axis=0)
        embedding = SpectralEmbedding(n_components=2)
        cxy = embedding.fit_transform(cxy)
        # rescale the coordinates
        cxy = (cxy - np.min(cxy, axis=0)) / (np.max(cxy, axis=0) - np.min(cxy, axis=0))
        cxy = cxy * (max_cxy - min_cxy) + min_cxy
        # print(cxy)

        for i, box in enumerate(self.canvas.iter_box()):
            box.set_center_xy(cxy[i, 0], cxy[i, 1])
    
    def init_plc_your_method(self):
        '''
        init the boxes to your method
        '''
        # TODO: your method
        pass

    # placement methods
    def triplet_loss_placement(self, margin = 1.0, iteration = 100, verbose = False):
        '''
        randomly choose two different nets: A and B
        Get A's source box and sink box
        Get B's source box and sink box
        Choose A's source box as anchor, A's sink box as positive, B's sink box as negative
        Compute the triplet loss and backpropagate to update the placement of the boxes
        Repeat for iteration times
        '''
        # randomly choose two different nets: A and B
        for i in tqdm(range(iteration)):
            net_A = np.random.choice(list(self.canvas.nets.values()))
            net_B = np.random.choice(list(self.canvas.nets.values()))
            while net_A.net_name == net_B.net_name:
                net_B = np.random.choice(list(self.canvas.nets.values()))
            print("Iteration %d: %s, %s" % (i, net_A.net_name, net_B.net_name)) if verbose else None
            # Get A's source box and sink box
            net_A_box_cxy = self.canvas.get_net_box_cxy(net_A.net_name)
            source_box_A_cxy = net_A_box_cxy[0]
            sink_box_A_cxy = net_A_box_cxy[1]
            # Get B's source box and sink box
            net_B_box_cxy = self.canvas.get_net_box_cxy(net_B.net_name)
            source_box_B_cxy = net_B_box_cxy[0]
            sink_box_B_cxy = net_B_box_cxy[1]
            # Choose A's source box as anchor, A's sink box as positive, B's sink box as negative
            anchor = torch.tensor([source_box_A_cxy, source_box_A_cxy], requires_grad=True)
            positive = torch.tensor([sink_box_A_cxy, sink_box_A_cxy], requires_grad=True)
            negative = torch.tensor([sink_box_B_cxy, sink_box_B_cxy], requires_grad=True)
            # Compute the triplet loss and backpropagate to update the placement of the boxes
            optimizer = optim.SGD([anchor, positive, negative], lr=0.01)
            optimizer.zero_grad()
            print("     anchor: %s, positive: %s, negative: %s" % (anchor, positive, negative)) if verbose else None
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
            loss = triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            # update the placement of the boxes
            self.canvas.set_box_cxy(net_A.get_source_box_name(), anchor[0][0].item(), anchor[0][1].item())
            self.canvas.set_box_cxy(net_A.get_sink_box_name(1), anchor[1][0].item(), anchor[1][1].item())
            self.canvas.set_box_cxy(net_B.get_sink_box_name(1), negative[1][0].item(), negative[1][1].item())

            # save the placement plot
            self.canvas.plot(savefig=True, filename="./plot/placement_%d.png" % i)

    def your_placement_method(self):
        '''
        your placement method
        '''
        # TODO: your placement method
        pass
