import numpy as np
from matplotlib import pyplot as plt

# Torch
import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.nn import functional as F

# JAX
from jax import grad, value_and_grad
from jax import numpy as jnp

# pickle for saving&loading
import pickle
from sklearn.svm import SVC
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm_notebook as tqdm

# custom modules
from canvas import *


class PCBPlacer:
    def __init__(self, canvas: PCBCanvas):
        self.canvas = canvas

    # initialization methods
    def init_plc_center(self):
        """
        init the boxes to the center of the canvas
        """
        # init boxes
        for box in self.canvas.iter_box():
            box.set_center_xy(self.canvas.width / 2, self.canvas.height / 2)

    def init_plc_spectral(self):
        """
        init the boxes to the spectral embedding of the canvas
        """
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
        """
        init the boxes to your method
        """
        # TODO: your method
        pass

    # placement methods
    def triplet_loss_placement(self, margin=1.0, iteration=100, verbose=False):
        """
        randomly choose two different nets: A and B
        Get A's source box and sink box
        Get B's source box and sink box
        Choose A's source box as anchor, A's sink box as positive, B's sink box as negative
        Compute the triplet loss and backpropagate to update the placement of the boxes
        Repeat for iteration times
        """
        # randomly choose two different nets: A and B
        for i in tqdm(range(iteration)):
            net_A = np.random.choice(list(self.canvas.nets.values()))
            net_B = np.random.choice(list(self.canvas.nets.values()))
            while net_A.net_name == net_B.net_name:
                net_B = np.random.choice(list(self.canvas.nets.values()))
            print(
                "Iteration %d: %s, %s" % (i, net_A.net_name, net_B.net_name)
            ) if verbose else None
            # Get A's source box and sink box
            net_A_box_cxy = self.canvas.get_net_box_cxy(net_A.net_name)
            source_box_A_cxy = net_A_box_cxy[0]
            sink_box_A_cxy = net_A_box_cxy[1]
            # Get B's source box and sink box
            net_B_box_cxy = self.canvas.get_net_box_cxy(net_B.net_name)
            source_box_B_cxy = net_B_box_cxy[0]
            sink_box_B_cxy = net_B_box_cxy[1]
            # Choose A's source box as anchor, A's sink box as positive, B's sink box as negative
            anchor = torch.tensor(
                [source_box_A_cxy, source_box_A_cxy], requires_grad=True
            )
            positive = torch.tensor(
                [sink_box_A_cxy, sink_box_A_cxy], requires_grad=True
            )
            negative = torch.tensor(
                [sink_box_B_cxy, sink_box_B_cxy], requires_grad=True
            )
            # Compute the triplet loss and backpropagate to update the placement of the boxes
            optimizer = optim.SGD([anchor, positive, negative], lr=0.01)
            optimizer.zero_grad()
            print(
                "     anchor: %s, positive: %s, negative: %s"
                % (anchor, positive, negative)
            ) if verbose else None
            triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
            loss = triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            # update the placement of the boxes
            self.canvas.set_box_cxy(
                net_A.get_source_box_name(), anchor[0][0].item(), anchor[0][1].item()
            )
            self.canvas.set_box_cxy(
                net_A.get_sink_box_name(1), anchor[1][0].item(), anchor[1][1].item()
            )
            self.canvas.set_box_cxy(
                net_B.get_sink_box_name(1), negative[1][0].item(), negative[1][1].item()
            )
            # self.canvas.move_all_boxes()

            # save the placement plot
            self.canvas.plot(
                savefig=True, filename="./plot/triplet/placement_%d.png" % i
            )

    def wirelength_objective_fn(self):
        """
        compute the wirelength objective function
        """
        wirelength = 0
        for net in self.canvas.iter_net():
            net_box_cxy = self.canvas.get_net_box_cxy(net.net_name)
            source_box_cxy = net_box_cxy[0]
            sink_box_cxy = net_box_cxy[1]
            wirelength += np.linalg.norm(source_box_cxy - sink_box_cxy)
        return wirelength

    def wirelength_objective_fn(self, net_to_pin_coords, net_to_insts, verbose=False):
        """
        compute the wirelength objective function
        """
        wirelength = 0
        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = np.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    # compute the weighted average wirelength
                    cur_WA = np.linalg.norm(pin_coords[i] - pin_coords[j])
                    print(
                        "     WA obj for net {} between pin {} and {}: {}".format(
                            key, i, j, cur_WA
                        )
                    ) if verbose else 0
                    wirelength += cur_WA
        return wirelength

    def wirelength_objective_fn_jax(
        self, net_to_pin_coords, net_to_insts, verbose=False
    ):
        """
        compute the wirelength objective function
        """
        wirelength = 0
        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = jnp.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    # compute the weighted average wirelength
                    cur_WA = jnp.linalg.norm(pin_coords[i] - pin_coords[j])
                    print(
                        "     WA obj for net {} between pin {} and {}: {}".format(
                            key, i, j, cur_WA
                        )
                    ) if verbose else 0
                    wirelength += cur_WA
        return wirelength

    def wirelength_objective_fn_grad(
        self, net_to_pin_coords, net_to_insts, verbose=False
    ):
        """
        use jax to compute the gradient of the wirelength objective function
        """
        # auxiliary variables P
        P_inst_grad = {}

        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = jnp.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    __, g = value_and_grad(np.linalg.norm, argnums=(0))(
                        pin_coords[i] - pin_coords[j]
                    )
                    print(
                        "     WA obj grad for net {} between pin {} and {}: {}".format(
                            key, i, j, g
                        )
                    ) if verbose else 0
                    # for each pin, add the gradient to the corresponding instance
                    for pin_i, (pin_coord, inst) in enumerate(
                        zip(
                            [pin_coords[i], pin_coords[j]],
                            [inst_value[i], inst_value[j]],
                        )
                    ):
                        if inst in P_inst_grad:
                            P_inst_grad[inst] += g[pin_i]
                        else:
                            P_inst_grad[inst] = g[pin_i]

        # take average of the gradients for each instance
        for key, value in P_inst_grad.items():
            P_inst_grad[key] = value / len(net_to_pin_coords[key])

        return P_inst_grad

    def wirelength_objective_fn_grad_jax(
        self, net_to_pin_coords, net_to_insts, verbose=False
    ):
        """
        use jax to compute the gradient of the wirelength objective function
        """
        # auxiliary variables P
        P_inst_grad = {}

        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = jnp.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    __, g = value_and_grad(jnp.linalg.norm, argnums=(0))(
                        pin_coords[i] - pin_coords[j]
                    )
                    print(
                        "     WA obj grad for net {} between pin {} and {}: {}".format(
                            key, i, j, g
                        )
                    ) if verbose else 0
                    # for each pin, add the gradient to the corresponding instance
                    for pin_i, (pin_coord, inst) in enumerate(
                        zip(
                            [pin_coords[i], pin_coords[j]],
                            [inst_value[i], inst_value[j]],
                        )
                    ):
                        print("inst: ", inst) if verbose else 0
                        if inst in P_inst_grad:
                            P_inst_grad[inst] += g[pin_i]
                        else:
                            P_inst_grad[inst] = g[pin_i]

        # take average of the gradients for each instance
        for key, value in P_inst_grad.items():
            P_inst_grad[key] = value / len(net_to_pin_coords[key])

        return P_inst_grad

    def density_objective_fn(self, bin_x=10, bin_y=10):
        """
        compute the density objective function
        """
        # compute the density of each bin
        density = np.zeros((bin_x, bin_y))
        for box in self.canvas.iter_box():
            # compute the bin index
            bin_index_x = int(box.cx / self.canvas.width * bin_x)
            bin_index_y = int(box.cy / self.canvas.height * bin_y)
            # update the density
            density[bin_index_x, bin_index_y] += 1
        # compute the density objective function
        density_objective = 0
        for i in range(bin_x):
            for j in range(bin_y):
                density_objective += np.abs(density[i, j] - np.mean(density))
        return density_objective

    def objective_fn(self, density_lambda=0.5):
        """
        compute the objective function
        """
        return (
            self.wirelength_objective_fn()
            + density_lambda * self.density_objective_fn()
        )

    def gradient_descent_placement(
        self, iteration=100, density_lambda=0.5, verbose=False
    ):
        """
        compute the gradient of the objective function
        update the placement of the boxes
        """
        # compute the gradient of the objective function
        for i in tqdm(range(iteration)):
            # compute the gradient of the objective function
            for box in self.canvas.iter_box():
                # compute the gradient of the wirelength objective function
                wirelength_gradient = 0
                for net in self.canvas.iter_net():
                    net_box_cxy = self.canvas.get_net_box_cxy(net.net_name)
                    source_box_cxy = net_box_cxy[0]
                    sink_box_cxy = net_box_cxy[1]
                    if box.box_name == net.get_source_box_name():
                        wirelength_gradient += (
                            source_box_cxy - sink_box_cxy
                        ) / np.linalg.norm(source_box_cxy - sink_box_cxy)
                    if box.box_name == net.get_sink_box_name(1):
                        wirelength_gradient += (
                            sink_box_cxy - source_box_cxy
                        ) / np.linalg.norm(sink_box_cxy - source_box_cxy)
                # compute the gradient of the density objective function
                density_gradient = 1
                # compute the gradient of the objective function
                gradient = wirelength_gradient + density_lambda * density_gradient
                # update the placement of the boxes
                box.set_center_xy(box.cx - gradient[0], box.cy - gradient[1])
            # save the placement plot
            self.canvas.plot(
                savefig=True, filename="./plot/plot04/placement_%d.png" % i
            )

    def Wa_obj(self, pin_mat, SMOOTHNESS=1.0):
        """Weighted Average Wirelength
        X: n x 2 matrix of x,y coordinates
        """
        x_term1_de = jnp.sum(pin_mat[:, 0] * jnp.exp(pin_mat[:, 0] / SMOOTHNESS))
        x_term1_nu = jnp.sum(jnp.exp(pin_mat[:, 0] / SMOOTHNESS))
        x_term2_de = jnp.sum(pin_mat[:, 0] * jnp.exp(-pin_mat[:, 0] / SMOOTHNESS))
        x_term2_nu = jnp.sum(jnp.exp(-pin_mat[:, 0] / SMOOTHNESS))

        y_term1_de = jnp.sum(pin_mat[:, 1] * jnp.exp(pin_mat[:, 1] / SMOOTHNESS))
        y_term1_nu = jnp.sum(jnp.exp(pin_mat[:, 1] / SMOOTHNESS))
        y_term2_de = jnp.sum(pin_mat[:, 1] * jnp.exp(-pin_mat[:, 1] / SMOOTHNESS))
        y_term2_nu = jnp.sum(jnp.exp(-pin_mat[:, 1] / SMOOTHNESS))

        x_term1 = x_term1_de / x_term1_nu
        x_term2 = x_term2_de / x_term2_nu
        y_term1 = y_term1_de / y_term1_nu
        y_term2 = y_term2_de / y_term2_nu

        return x_term1 - x_term2 + y_term1 - y_term2

    def Wa_obj_grad(self, net_to_pin_coords, net_to_insts, SMOOTHNESS=1.0):
        Wa_sum = 0
        pin_grad_to_inst = {}  # aggregate the pin gradients for each instance
        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # get the pin coordinates
            pin_coords = jnp.array(pin_value)
            # compute the weighted average wirelength only between the first pin and the rest
            for i in range(1, len(pin_coords)):
                # compute the weighted average wirelength
                v, g = value_and_grad(self.Wa_obj)(
                    np.array([pin_coords[0], pin_coords[i]]), SMOOTHNESS=SMOOTHNESS
                )
                Wa_sum += v
                # for each pin, add the gradient to the corresponding instance
                for pin_i, (pin_coord, inst) in enumerate(
                    zip([pin_coords[0], pin_coords[i]], [inst_value[0], inst_value[i]])
                ):
                    if inst in pin_grad_to_inst:
                        pin_grad_to_inst[inst] += g[pin_i]
                    else:
                        pin_grad_to_inst[inst] = g[pin_i]
        return pin_grad_to_inst

    # use HPWL as the objective function instead of WA
    def HPWL_obj(self, pin_mat):
        """Half Perimeter Wirelength
        X: n x 2 matrix of x,y coordinates
        """
        return (
            jnp.max(pin_mat[:, 0])
            - jnp.min(pin_mat[:, 0])
            + jnp.max(pin_mat[:, 1])
            - jnp.min(pin_mat[:, 1])
        )

    def HPWL_obj_grad(self, net_to_pin_coords, net_to_insts, verbose=False):
        """Weighted Average Wirelength
        X: n x 2 matrix of x,y coordinates
        """
        # auxiliary variables P
        P_inst_grad = {}

        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = jnp.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    __, g = value_and_grad(self.HPWL_obj, argnums=(0))(
                        np.array([pin_coords[i], pin_coords[j]])
                    )
                    print(
                        "     HPWL obj grad for net {} between pin {} and {}: {}".format(
                            key, i, j, g
                        )
                    ) if verbose else 0
                    # for each pin, add the gradient to the corresponding instance
                    for pin_i, (pin_coord, inst) in enumerate(
                        zip(
                            [pin_coords[i], pin_coords[j]],
                            [inst_value[i], inst_value[j]],
                        )
                    ):
                        if inst in P_inst_grad:
                            P_inst_grad[inst] += g[pin_i]
                        else:
                            P_inst_grad[inst] = g[pin_i]

        return P_inst_grad

    ############################
    #    Density Objective     #
    ############################
    def Density_obj(
        self,
        compx_,
        compy_,
        compw_,
        comph_,
        canvas_dims,
        num_bin_x,
        num_bin_y,
        verbose=False,
    ):
        """Density Objective
        X: n x 2 matrix of x,y coordinates
        canvas_dims: tuple of (maxX, maxY, minX, minY)
        """
        # get a vector of x and y coordinates of all boxes
        width = canvas_dims[0] - canvas_dims[2]
        height = canvas_dims[1] - canvas_dims[3]

        # get bin size
        bin_size_x = width / num_bin_x
        bin_size_y = height / num_bin_y

        # num of bins
        num_bin = num_bin_x * num_bin_y

        # num of components
        num_comp = len(compx_)

        # x coordinates of bins
        bin_x = jnp.array(
            [
                canvas_dims[2] + (j % num_bin_x) * bin_size_x + bin_size_x / 2
                for j in range(num_bin)
            ]
        )
        # y coordinates of bins
        bin_y = jnp.array(
            [
                canvas_dims[3] + (j // num_bin_x) * bin_size_y + bin_size_y / 2
                for j in range(num_bin)
            ]
        )

        # width and height of bins
        bin_width = jnp.array([bin_size_x for j in range(num_bin)])
        bin_height = jnp.array([bin_size_y for j in range(num_bin)])

        print("bin_x: ", bin_x) if verbose else 0
        print("bin_y: ", bin_y) if verbose else 0

        # get d_x and d_y
        # NOTE: compx and compy are the bottom left coordinates of the components
        d_x = jnp.abs(jnp.repeat(compx_, num_bin) - jnp.tile(bin_x, len(compx_)))
        d_y = jnp.abs(jnp.repeat(compy_, num_bin) - jnp.tile(bin_y, len(compy_)))

        print(
            "jnp.repeat(compx_, num_bin) shape", jnp.repeat(compx_, num_bin).shape
        ) if verbose else 0
        print(
            "jnp.repeat(compx_, num_bin)", jnp.repeat(compx_, num_bin)
        ) if verbose else 0
        print(
            "jnp.tile(bin_x, len(compx_)) shape", jnp.tile(bin_x, len(compx_)).shape
        ) if verbose else 0
        print(
            "jnp.tile(bin_x, len(compx_))", jnp.tile(bin_x, len(compx_))
        ) if verbose else 0

        print("d_x: ", d_x.shape) if verbose else 0
        print("d_x: ", d_x) if verbose else 0
        print("d_y: ", d_y.shape) if verbose else 0
        print("d_y: ", d_y) if verbose else 0

        # tile bin_width and bin_height, lines up with d_x and d_y
        bin_width_tile = jnp.tile(bin_width, len(compx_))
        bin_height_tile = jnp.tile(bin_height, len(compy_))

        print("bin_width_tile: ", bin_width_tile.shape) if verbose else 0
        print("bin_width_tile: ", bin_width_tile) if verbose else 0
        print("bin_height_tile: ", bin_height_tile.shape) if verbose else 0
        print("bin_height_tile: ", bin_height_tile) if verbose else 0

        epsilon = 1e-5  # prevent C having infinity values

        # compute theta_x and theta_y
        theta_x = jnp.where(
            (0 <= d_x) & (d_x <= bin_width_tile / 2),
            1 - 2 * d_x**2 / bin_width_tile**2,
            jnp.where(
                (bin_width_tile / 2 <= d_x) & (d_x <= bin_width_tile),
                (2 * ((d_x - bin_width_tile) ** 2)) / (bin_width_tile**2),
                epsilon,
            ),
        )

        # assert there are no nan values
        try:
            assert not jnp.isnan(theta_x).any()
        except AssertionError:
            print("[ERROR] theta_x has nan values")
            print(theta_x)

        theta_y = jnp.where(
            (0 <= d_y) & (d_y <= bin_height_tile / 2),
            1 - 2 * d_y**2 / bin_height_tile**2,
            jnp.where(
                (bin_height_tile / 2 <= d_y) & (d_y <= bin_height_tile),
                (2 * ((d_y - bin_height_tile) ** 2)) / (bin_height_tile**2),
                epsilon,
            ),
        )

        # assert there are no nan values
        try:
            assert not jnp.isnan(theta_y).any()
        except AssertionError:
            print("[ERROR] theta_y has nan values")
            print(theta_y)

        print("theta_x: ", theta_x.shape) if verbose else 0
        print(theta_x) if verbose else 0
        print("theta_y: ", theta_y.shape) if verbose else 0
        print(theta_y) if verbose else 0

        # select theta_x with respect to each component
        # NOTE: this is just reshaping theta_x and theta_y to (num_comp, num_bin)
        theta_x_comp = jnp.array(
            [theta_x[i * num_bin : (i + 1) * num_bin] for i in range(len(compx_))]
        )

        print("theta_x_comp: ", theta_x_comp.shape) if verbose else 0

        # select theta_y with respect to each component
        theta_y_comp = jnp.array(
            [theta_y[i * num_bin : (i + 1) * num_bin] for i in range(len(compy_))]
        )

        print("theta_y_comp: ", theta_y_comp.shape) if verbose else 0

        # compute a vec of component area
        comp_area = jnp.array(compw_) * jnp.array(comph_)

        print(
            "jnp.sum(theta_x_comp * theta_y_comp, axis=1): ",
            jnp.sum(theta_x_comp * theta_y_comp, axis=1).shape,
        ) if verbose else 0

        # for each component, compute C_i = (w_i * h_i) / sum_b (theta_x * theta_y)
        # From the perspective of bins, choose theta based on instances I overlapped with
        # Each instance has their own C, so C should be a vector in R^8
        C = comp_area / jnp.sum(theta_x_comp * theta_y_comp, axis=1)

        print("C: ", C.shape) if verbose else 0
        print("C: ", C) if verbose else 0

        # if there is any inifity, print its index and value
        if jnp.isinf(C).any():
            print("[ERROR] C has infinity values")
            print(C)
            print("index: ", jnp.where(jnp.isinf(C))) if verbose else 0
            print("value: ", C[jnp.where(jnp.isinf(C))]) if verbose else 0

            # using the index of infinity, print the comp_area and jnp.sum(theta_x_comp * theta_y_comp, axis=1)
            print("comp_area: ", comp_area[jnp.where(jnp.isinf(C))]) if verbose else 0
            print(
                "sum: ",
                jnp.sum(theta_x_comp * theta_y_comp, axis=1)[jnp.where(jnp.isinf(C))],
            ) if verbose else 0

        # transform C to a matrix with (num_bin, num_comp)
        C_mat = jnp.tile(C, num_bin).reshape(num_bin, num_comp)

        print("C_mat: ", C_mat.shape) if verbose else 0
        print(
            "C_mat: ", C_mat
        ) if verbose else 0  # line up with comp, so this is correct

        # select theta_x and theta_y with respect to each bin (every # component)
        # NOTE: this is just reshaping theta_x and theta_y to (num_bin, num_comp)
        theta_x_bin = jnp.array([theta_x[i::num_bin] for i in range(num_bin)])

        theta_y_bin = jnp.array([theta_y[i::num_bin] for i in range(num_bin)])

        print("theta_x_bin: ", theta_x_bin.shape) if verbose else 0
        print("theta_y_bin: ", theta_y_bin.shape) if verbose else 0

        # compute density
        density_bin = jnp.sum(C_mat * theta_x_bin * theta_y_bin, axis=1)
        print(
            "density_bin: ", density_bin.shape
        ) if verbose else 0  # should be over all bins
        print("density_bin: ", density_bin) if verbose else 0

        density_comp = jnp.sum(comp_area) / num_bin
        density_comp = jnp.tile(density_comp, num_bin)
        print("density_comp: ", density_comp.shape) if verbose else 0
        print("density_comp: ", density_comp) if verbose else 0
        # tile density_comp to match density_bin

        total_density = jnp.sum(jnp.power(density_bin - density_comp, 2))

        print("total_density: ", total_density) if verbose else 0

        return total_density

    def Density_obj_grad(
        self,
        compx_,
        compy_,
        compw_,
        comph_,
        canvas_dims,
        num_bin_x,
        num_bin_y,
        verbose=False,
    ):
        __, (Density_grad_x, Density_grad_y) = value_and_grad(
            self.Density_obj, argnums=(0, 1)
        )(
            compx_,
            compy_,
            compw_,
            comph_,
            canvas_dims,
            num_bin_x,
            num_bin_y,
            verbose=False,
        )

        P_inst_grad = {}

        for i, (grad_x, grad_y) in enumerate(zip(Density_grad_x, Density_grad_y)):
            box_name = f"box_{i}"
            if box_name in P_inst_grad:
                P_inst_grad[box_name] += np.array([grad_x, grad_y])
            else:
                P_inst_grad[box_name] = np.array([grad_x, grad_y])

        return P_inst_grad


    def Grid_density_obj(
        self,
        compx_,
        compy_,
        compw_,
        comph_,
        canvas_dims,
        num_bin_x,
        num_bin_y,
        verbose=False,
    ):
        '''
        compute the density objective function
        - For each grid bin,
            - gather all the components that overlap with the bin
            - compute the max density of the bin by summing up the area of the components
        - sum up the max density of all the bins
        '''

        def check_overlap(bin_x, bin_y, bin_size_x, bin_size_y, comp_x, comp_y, comp_w, comp_h):
            """Check if the component overlaps with the bin"""
            # check if the component overlaps with the bin
            if (
                comp_x + comp_w / 2 < bin_x - bin_size_x / 2
                or comp_x - comp_w / 2 > bin_x + bin_size_x / 2
            ):
                return False
            if (
                comp_y + comp_h / 2 < bin_y - bin_size_y / 2
                or comp_y - comp_h / 2 > bin_y + bin_size_y / 2
            ):
                return False
            return True

        def compute_overlap_area(bin_x, bin_y, bin_size_x, bin_size_y, comp_x, comp_y, comp_w, comp_h):
            """Compute the intersection area between the bin and the component"""
            # compute the intersection area between the bin and the component
            dx, dy = 0.0, 0.0
            dx = min(bin_x + bin_size_x / 2, comp_x + comp_w / 2) - max(
                bin_x - bin_size_x / 2, comp_x - comp_w / 2
            )
            dy = min(bin_y + bin_size_y / 2, comp_y + comp_h / 2) - max(
                bin_y - bin_size_y / 2, comp_y - comp_h / 2
            )
            return dx * dy

        # get a vector of x and y coordinates of all boxes
        width = canvas_dims[0] - canvas_dims[2]
        height = canvas_dims[1] - canvas_dims[3]

        # get bin size
        bin_size_x = width / num_bin_x
        bin_size_y = height / num_bin_y

        # num of bins
        num_bin = num_bin_x * num_bin_y

        # num of components
        num_comp = len(compx_)

        # x coordinates of bins
        bin_x = jnp.array(
            [
                canvas_dims[2] + (j % num_bin_x) * bin_size_x + bin_size_x / 2
                for j in range(num_bin)
            ]
        )

        # y coordinates of bins
        bin_y = jnp.array(
            [
                canvas_dims[3] + (j // num_bin_x) * bin_size_y + bin_size_y / 2
                for j in range(num_bin)
            ]
        )

        bin_density = jnp.array([0.0 for j in range(num_bin)])

        # gather all the components that overlap with the bin
        for bin_i, (bin_x_i, bin_y_i) in enumerate(zip(bin_x, bin_y)):
            for comp_i, (comp_x_i, comp_y_i, comp_w_i, comp_h_i) in enumerate(zip(compx_, compy_, compw_, comph_)):
                # check if the component overlaps with the bin
                if check_overlap(bin_x_i, bin_y_i, bin_size_x, bin_size_y, comp_x_i, comp_y_i, comp_w_i, comp_h_i):
                    bin_density = bin_density.at[bin_i].add(compute_overlap_area(bin_x_i, bin_y_i, bin_size_x, bin_size_y, comp_x_i, comp_y_i, comp_w_i, comp_h_i))

        # divide the bin density by the bin size
        bin_density = bin_density / (bin_size_x * bin_size_y)

        # compute the max density
        max_density = jnp.max(bin_density)

        return max_density

    
    def Grid_density_obj_grad(
        self,
        compx_,
        compy_,
        compw_,
        comph_,
        canvas_dims,
        num_bin_x,
        num_bin_y,
        verbose=False,
    ):
        __, (g_density_grad_x, g_density_grad_y) = value_and_grad(
            self.Grid_density_obj, argnums=(0, 1)
        )(
            compx_,
            compy_,
            compw_,
            comph_,
            canvas_dims,
            num_bin_x,
            num_bin_y,
            verbose=False,
        )

        P_inst_grad = {}

        for i, (grad_x, grad_y) in enumerate(zip(g_density_grad_x, g_density_grad_y)):
            box_name = f"box_{i}"
            if box_name in P_inst_grad:
                P_inst_grad[box_name] += np.array([grad_x, grad_y])
            else:
                P_inst_grad[box_name] = np.array([grad_x, grad_y])

        return P_inst_grad



    def check_boundary(
        self,
        curr_x,
        curr_y,
        width,
        height,
        delta_x,
        delta_y,
        min_x,
        min_y,
        max_x,
        max_y,
    ):
        """Check if the next position is within the boundary"""
        # check if the next position is within the boundary
        if curr_x + delta_x + width / 2 > max_x or curr_x + delta_x - width / 2 < min_x:
            # print("     curr_x + delta_x + width / 2 vs maxx:", curr_x + delta_x + width / 2, max_x)
            # print("     curr_x + delta_x - width / 2 vs minx:", curr_x + delta_x - width / 2, min_x)
            delta_x = 0
        if (
            curr_y + delta_y + height / 2 > max_y
            or curr_y + delta_y - height / 2 < min_y
        ):
            # print("     curr_y + delta_y + height / 2 vs maxy:", curr_y + delta_y + height / 2, max_y)
            # print("     curr_y + delta_y - height / 2 vs miny:", curr_y + delta_y - height / 2, min_y)
            delta_y = 0
        return delta_x, delta_y

    # def equation_2(self, compx_jax, compy_jax, compw, comph, canvas_dims, num_bin_x, num_bin_y, net_to_pin_coords, U, r, smooth, verbose=False):
    def equation_2(
        self,
        compx_jax,
        compy_jax,
        compw,
        comph,
        canvas_dims,
        num_bin_x,
        num_bin_y,
        net_to_pin_coords,
        net_to_insts,
        smooth,
        verbose=False,
    ):
        """Equation 2 in the paper"""
        Wa_sum = 0
        Density_sum = 0

        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = np.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    # compute the weighted average wirelength
                    cur_WA = self.Wa_obj(
                        np.array([pin_coords[i], pin_coords[j]]), smooth
                    )
                    print(
                        "     WA obj for net {} between pin {} and {}: {}".format(
                            key, i, j, cur_WA
                        )
                    ) if verbose else 0
                    Wa_sum += cur_WA

        # assert Wa_sum is not NaN
        try:
            assert not np.isnan(Wa_sum)
        except AssertionError:
            print("[ERROR] Wa_sum is NaN")

        # compute the density objective
        Density_sum = self.Density_obj(
            compx_jax,
            compy_jax,
            compw,
            comph,
            canvas_dims,
            num_bin_x,
            num_bin_y,
            verbose=False,
        )

        # assert Density_sum is not NaN
        try:
            assert not np.isnan(Density_sum)
        except AssertionError:
            print("[ERROR] Density_sum is NaN")
            # save the current state
            print("Saving current state...")
            with open("error_current_state.pkl", "wb") as f:
                pickle.dump(
                    [
                        compx_jax,
                        compy_jax,
                        compw,
                        comph,
                        canvas_dims,
                        num_bin_x,
                        num_bin_y,
                        net_to_pin_coords,
                        U,
                        r,
                        smooth,
                    ],
                    f,
                )

        # return Wa_sum + lambda_D * Density_sum + lambda_NS * NS_sum
        return Wa_sum, Density_sum

    # def equation_2_grad(self, compx_jax, compy_jax, compw, comph, canvas_dims, num_bin_x, num_bin_y, net_to_pin_coords, U, r, smooth, fixed_idx=None, verbose=False):
    def equation_2_grad(
        self,
        compx_jax,
        compy_jax,
        compw,
        comph,
        canvas_dims,
        num_bin_x,
        num_bin_y,
        net_to_pin_coords,
        net_to_insts,
        smooth,
        fixed_idx=None,
        verbose=False,
    ):
        """Equation 2 in the paper"""
        # auxiliary variables P
        Wa_pin_grad_to_inst = {}

        for key, pin_value, inst_value in zip(
            net_to_pin_coords.keys(), net_to_pin_coords.values(), net_to_insts.values()
        ):
            # compute the weighted average wirelength
            # get the pin coordinates
            pin_coords = np.array(pin_value)
            # compute the weighted average wirelength between each pair of pins
            for i in range(len(pin_coords) - 1):
                for j in range(i + 1, len(pin_coords)):
                    __, g = value_and_grad(self.Wa_obj, argnums=(0))(
                        np.array([pin_coords[i], pin_coords[j]]), smooth
                    )
                    print(
                        "     WA obj grad for net {} between pin {} and {}: {}".format(
                            key, i, j, g
                        )
                    ) if verbose else 0
                    # for each pin, add the gradient to the corresponding instance
                    for pin_i, (pin_coord, inst) in enumerate(
                        zip(
                            [pin_coords[i], pin_coords[j]],
                            [inst_value[i], inst_value[j]],
                        )
                    ):
                        if inst in Wa_pin_grad_to_inst:
                            Wa_pin_grad_to_inst[inst] += g[pin_i]
                        else:
                            Wa_pin_grad_to_inst[inst] = g[pin_i]

        # compute the density objective
        __, (Density_grad_x, Density_grad_y) = value_and_grad(
            self.Density_obj, argnums=(0, 1)
        )(
            compx_jax,
            compy_jax,
            compw,
            comph,
            canvas_dims,
            num_bin_x,
            num_bin_y,
            verbose=False,
        )
        print("     Density_grad_x: ", Density_grad_x) if verbose else 0
        print("     Density_grad_y: ", Density_grad_y) if verbose else 0

        # # take average of the gradients for each instance
        # for key, value in P_inst_grad.items():
        #     P_inst_grad[key] = value / len(net_to_pin_coords[key])

        return (Density_grad_x, Density_grad_y), Wa_pin_grad_to_inst

    def your_placement_method(self):
        """
        your placement method
        """
        # TODO: your placement method
        pass
