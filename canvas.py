import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from sklearn.manifold import SpectralEmbedding
import re
import os
import imageio
import pickle 

# custom modules
from box import Box
from net import Net

# arcface loss
from pytorch_metric_learning import losses, testers


class PCBCanvas:
    def __init__(self, width: float, height: float, boxes: dict, nets: dict):
        self.width = width
        self.height = height
        # entities
        self.boxes = boxes
        self.nets = nets
        self.num_boxes = len(boxes)
        self.num_nets = len(nets)

    def get_canvas_dim(self):
        """
        return (maxX, maxY, minX, minY)
        """
        return (self.width, self.height, 0, 0)

    def add_box(self, box):
        """
        add a box to the canvas
        """
        self.boxes[box.box_name] = box
        # update num_boxes
        self.num_boxes += 1

    def get_box_by_name(self, box_name):
        """
        return the box by name
        """
        return self.boxes[box_name]

    def iter_box(self):
        """
        iterate through the boxes
        """
        for box in self.boxes.values():
            yield box

    def add_net(self, net):
        """
        add a net to the canvas
        """
        self.nets[net.net_name] = net
        # assert that the source box name is in the canvas
        source_box_name = net.boxes[0].box_name
        for box in self.iter_box():
            if box.box_name == source_box_name:
                break
        else:
            print("[ERROR] Source box %s not found in canvas" % source_box_name)
            exit(1)
        # assert that the sink box names are in the canvas
        for box in net.iter_box():
            box_name = box.box_name
            for box in self.iter_box():
                if box.box_name == box_name:
                    break
            else:
                print("[ERROR] Sink box %s not found in canvas" % box_name)
                exit(1)

        # update num_nets
        self.num_nets += 1

    def iter_net(self):
        """
        iterate through the nets
        """
        for net in self.nets.values():
            yield net

    def iter_net_box(self):
        """
        iterate through the boxes in the nets
        """
        for net in self.iter_net():
            box_list = lambda net: [
                self.get_box_by_name(box_name) for box_name in net.iter_box_name()
            ]
            yield box_list(net)

    def get_net_box_names(self, net_name):
        """
        return a list of box names in the net
        """
        net = self.nets[net_name]
        box_names = []
        for box_name in net.iter_box_name():
            box_names.append(box_name)
        return box_names

    def get_net_box_cxy(self, net_name):
        """
        return a list of (cx, cy) of the boxes in the net
        """
        net = self.nets[net_name]
        cxy_list = []
        for box_name in net.iter_box_name():
            box = self.boxes[box_name]
            cxy_list.append((box.cx, box.cy))
        return np.array(cxy_list)

    def get_box_cxy(self, box_name):
        """
        return the center of the box
        """
        box = self.boxes[box_name]
        return np.array([box.cx, box.cy])

    def get_cxy(self):
        """
        return a list of (cx, cy) of the boxes in the canvas
        """
        cxy_list = []
        for box in self.iter_box():
            cxy_list.append((box.cx, box.cy))
        return np.array(cxy_list)

    def get_box_width(self, box_name):
        """
        return the width of the box
        """
        box = self.boxes[box_name]
        return box.width

    def get_box_height(self, box_name):
        """
        return the height of the box
        """
        box = self.boxes[box_name]
        return box.height

    def get_all_boxes_width(self):
        """
        return a list of the width of the boxes
        """
        width_list = []
        for box in self.iter_box():
            width_list.append(box.width)
        return width_list

    def get_all_boxes_height(self):
        """
        return a list of the height of the boxes
        """
        height_list = []
        for box in self.iter_box():
            height_list.append(box.height)
        return height_list

    def set_box_cxy(self, box_name, cx, cy):
        """
        set the center of the box
        """
        box = self.boxes[box_name]
        box.set_center_xy(cx, cy)

    def move_box(self, box_name, if_decay=True):
        """
        move the box by dx and dy
        decay the motion
        """
        box = self.boxes[box_name]
        box.set_center_xy(box.cx + box.dx, box.cy + box.dy)
        # box.decay_motion()
        if if_decay:
            box.decay_motion()
        # box.self_rotate()

    def move_all_boxes(self):
        """
        move all the boxes
        """
        for box in self.iter_box():
            self.move_box(box.box_name, if_decay=True)
            # only net spring force placement
            # self.net_spring_force_placement(k=0.1, l0=0.1, kd=0.1)
            self.kd_tree_based_boxes_collision_detection(scale=0.01)
        self.net_spring_force_placement(k=0.01, l0=5.0, kd=0.01)
        # self.boundary_collision_detection()
        # self.arcface_loss_between_nets(scale=0.01)
        self.boundary_collision_detection()

    def random_set_all_boxes_motion(self, max_dx, max_dy):
        """
        set the motion of the boxes randomly
        """
        for box in self.iter_box():
            rand_dx = np.random.uniform(-max_dx, max_dx)
            rand_dy = np.random.uniform(-max_dy, max_dy)
            print("Box %s: dx: %f, dy: %f" % (box.box_name, rand_dx, rand_dy))
            box.set_motion(
                box.dx + rand_dx,
                box.dy + rand_dy,
            )

    def kd_tree_based_boxes_collision_detection(self, scale=1.0):
        """
        detect the collision between boxes using kd tree
        need to consider the height and width of the boxes
        two boxes are considered to collide if their bounding boxes collide or overlap
        if two boxes collide, the boxes will bounce back with the scale
        """
        from scipy.spatial import KDTree

        # get the center of the boxes
        cxy = self.get_cxy()
        # get the height and width of the boxes
        hw = []
        for box in self.iter_box():
            hw.append([box.width, box.height])
        hw = np.array(hw)
        # build the kd tree
        tree = KDTree(cxy)
        # query the kd tree
        dist, ind = tree.query(cxy, k=2)

        # print(dist)
        # print(ind)

        # compute the bounding boxes
        bounding_boxes = []
        for i, box in enumerate(self.iter_box()):
            bounding_box = [
                box.cx - box.width / 2,
                box.cy - box.height / 2,
                box.cx + box.width / 2,
                box.cy + box.height / 2,
            ]
            bounding_boxes.append(bounding_box)
        bounding_boxes = np.array(bounding_boxes)

        # print(bounding_boxes)

        # compute the bounding boxes collision
        for i, box in enumerate(self.iter_box()):
            # get the bounding box of the box
            bounding_box = bounding_boxes[i]
            # get the bounding boxes of the neighbors
            bounding_box_neighbors = []
            for j in ind[i, 1:]:
                bounding_box_neighbors.append(bounding_boxes[j])
            bounding_box_neighbors = np.array(bounding_box_neighbors)
            # print(bounding_box_neighbors)
            # compute the bounding boxes collision
            for bounding_box_neighbor in bounding_box_neighbors:
                # print(bounding_box, bounding_box_neighbor)
                if (
                    bounding_box[0] < bounding_box_neighbor[2]
                    and bounding_box[2] > bounding_box_neighbor[0]
                    and bounding_box[1] < bounding_box_neighbor[3]
                    and bounding_box[3] > bounding_box_neighbor[1]
                ):
                    # print("Collision")
                    # compute the unit vector from the center of the box to the center of the neighbor
                    unit_vector = (cxy[ind[i, 1]] - cxy[i]) / np.linalg.norm(
                        cxy[ind[i, 1]] - cxy[i]
                    )
                    # print(unit_vector)
                    # compute the motion
                    dx = -unit_vector[0] * scale
                    dy = -unit_vector[1] * scale
                    # print(dx, dy)
                    # set the motion
                    box.set_motion(box.dx + dx, box.dy + dy)

    def arcface_loss_between_nets(self, scale=1.0):
        """
        compute the arcface loss between nets
        use pytorch_metric_learning to compute the loss
        then use the loss to compute the gradient
        then use the gradient to update the placement of the boxes
        """
        # get the center of the boxes
        cxy = self.get_cxy()
        # get the height and width of the boxes
        hw = []
        for box in self.iter_box():
            hw.append([box.width, box.height])
        hw = np.array(hw)
        # get the net names
        net_names = []
        for net in self.iter_net():
            net_names.append(net.net_name)
        net_names = np.array(net_names)
        # get the net box names
        net_box_names = []
        for net in self.iter_net():
            net_box_names.append(net.box_names)
        net_box_names = np.array(net_box_names)

        # print(cxy)
        # print(hw)

        # compute the arcface loss
        loss_func = losses.ArcFaceLoss()
        tester = testers.BaseTester()
        loss = loss_func.get_loss(
            tester,
            cxy,
            hw,
            net_names,
            net_box_names,
        )
        print(loss)

    def set_all_boxes_motion(self, dx_list, dy_list):
        """
        set the motion of the boxes
        """
        assert len(dx_list) == len(dy_list)
        assert len(dx_list) == self.num_boxes

        for i, box in enumerate(self.iter_box()):
            box.set_motion(dx_list[i], dy_list[i])

    def if_all_boxes_within_boundary(self):
        """
        check if all the boxes are within the boundary
        """
        for box in self.iter_box():
            if (
                box.llx < 0
                or box.urx > self.width
                or box.lly < 0
                or box.ury > self.height
            ):
                return False
        return True

    def net_spring_force_placement(self, k, l0, kd):
        """
        go through all the nets and compute the net spring force
        between the source box and sink boxes
        use Net.get_net_spring_force() to compute the net spring force
        spring force is applied to both the source box and sink boxes
        the source box will be affected by all the sink boxes
        """

        def get_velocity(box_name):
            """
            return the velocity of the box
            """
            box = self.boxes[box_name]
            return np.array([box.dx, box.dy])

        for net in self.iter_net():
            # get the source box
            source_box_name = net.get_source_box_name()
            source_box_cxy = self.get_box_cxy(source_box_name)
            # get the sink boxes
            sink_box_names = net.get_sink_box_names()
            sink_box_cxy = []
            for box_name in sink_box_names:
                box_cxy = self.get_box_cxy(box_name)
                sink_box_cxy.append(box_cxy)
            # compute the net spring force
            for i, box_cxy in enumerate(sink_box_cxy):
                F_net = Net.get_net_spring_force(
                    self,
                    source_box_cxy,
                    box_cxy,
                    get_velocity(source_box_name),
                    get_velocity(sink_box_names[i]),
                    k,
                    l0,
                    kd,
                )
                # print("[INFO] Net %s: F_net: %s" % (net.net_name, F_net))
                # apply the net spring force to the source box
                self.boxes[source_box_name].set_motion(
                    self.boxes[source_box_name].dx + F_net[0],
                    self.boxes[source_box_name].dy + F_net[1],
                )
                # apply the net spring force to the sink box
                self.boxes[sink_box_names[i]].set_motion(
                    self.boxes[sink_box_names[i]].dx - F_net[0],
                    self.boxes[sink_box_names[i]].dy - F_net[1],
                )

    def boundary_collision_detection(self, bounce_back_factor=1.0):
        """
        detect the collision with the boundary
        if a box collides with the boundary, the box will bounce back with the bounce_back_factor
        """
        for box in self.iter_box():
            # left boundary
            if box.llx < 0:
                box.set_motion(-box.llx * bounce_back_factor, 0)
            # right boundary
            if box.urx > self.width:
                box.set_motion((self.width - box.urx) * bounce_back_factor, 0)
            # bottom boundary
            if box.lly < 0:
                box.set_motion(0, -box.lly * bounce_back_factor)
            # top boundary
            if box.ury > self.height:
                box.set_motion(0, (self.height - box.ury) * bounce_back_factor)

    def get_incidence_matrix(self):
        """
        return the incidence matrix of the canvas
        """
        A = np.zeros((self.num_boxes, self.num_nets))
        for i, net in enumerate(self.iter_net()):
            for box_name in net.iter_box_name():
                # get box index in the self.boxes.keys()
                box_i = list(self.boxes.keys()).index(box_name)
                A[box_i, i] = 1
        return np.array(A)

    # metrics
    def get_total_wirelength_from_cxy(self):
        """
        return the total wirelength of the canvas
        """
        total_wirelength = 0
        cxy = self.get_cxy()
        # print(cxy)
        incidence_matrix = self.get_incidence_matrix()
        # print(incidence_matrix)
        # compute the total wirelength
        for i, net in enumerate(self.iter_net()):
            # get the source box
            source_box_name = net.get_source_box_name()
            source_box_cxy = self.get_box_cxy(source_box_name)
            # get the sink boxes
            sink_box_names = net.get_sink_box_names()
            sink_box_cxy = []
            for box_name in sink_box_names:
                box_cxy = self.get_box_cxy(box_name)
                sink_box_cxy.append(box_cxy)
            # compute the total wirelength
            for box_cxy in sink_box_cxy:
                total_wirelength += np.linalg.norm(source_box_cxy - box_cxy)

        return total_wirelength

    def canvas_info(self):
        """
        print the canvas info
        """
        print("[INFO] Canvas width: %d, height: %d" % (self.width, self.height))
        for box in self.iter_box():
            print(
                "     Box: %s llx: %d, lly: %d, width: %d, height: %d"
                % (box.box_name, box.llx, box.lly, box.width, box.height)
            )
        for net_box_list in self.iter_net_box():
            print("     Net: ", end="")
            for box in net_box_list:
                print("%s " % box.box_name, end="")
            print()

    def plot(self, savefig=False, filename="pcb.png"):
        """
        plot the canvas with boxes and nets
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.set_title("PCB")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # plot boxes
        for box in self.iter_box():
            if box.rotation == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (box.llx, box.lly),
                        box.width,
                        box.height,
                        facecolor="none",
                        edgecolor="black",
                    )
                )
            else:
                # use affine transformation to rotate the box
                t = Affine2D().rotate_deg_around(box.cx, box.cy, box.rotation)
                rect = plt.Rectangle(
                    (box.llx, box.lly),
                    box.width,
                    box.height,
                    facecolor="none",
                    edgecolor="black",
                )
                rect.set_transform(t + ax.transData)
                ax.add_patch(rect)
        # plot the nets
        for net_box_list in self.iter_net_box():
            x = []
            y = []
            # each line is from the center of the source box to the center of the sink box
            source_box_cx = net_box_list[0].cx
            source_box_cy = net_box_list[0].cy
            for box in net_box_list:
                ax.plot([source_box_cx, box.cx], [source_box_cy, box.cy], "b-")
        if savefig:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()

    def generate_gif_from_plots(self, plot_id, gifname="pcb.gif"):
        """
        generate a gif from the plot
        """

        def sort_key(s):
            if s:
                try:
                    c = re.findall(r"\d+$", s)[0]
                except:
                    c = -1
                return int(c)

        def sort_files(l):
            l.sort(key=sort_key)
            return l

        images = []
        plot_path = os.path.join("./plot", plot_id)
        filenames = sort_files(os.listdir(plot_path))
        filenames = [os.path.join(plot_path, filename) for filename in filenames]
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(gifname, images, duration=1.5)


import gymnasium as gym
from gymnasium import spaces


class PCB_Environment(gym.Env):
    metadata = {"render.modes": ["human", "graph"]}

    def __init__(self, canvas: PCBCanvas, time_limit: int) -> None:
        super(PCB_Environment, self).__init__()
        self.canvas = canvas
        self.time_limit = time_limit
        self.time_step = 0

        # action space: pick a box (box_name) and set the center of the box (cx, cy) as integer
        # (box_name_i, cx, cy)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.canvas.num_boxes, self.canvas.width, self.canvas.height]),
            dtype=np.int32,
        )

        # observation space: the center of the boxes (cx, cy), the width and height of the boxes (width, height), the incidence matrix of the canvas, the time step
        # cx, cy: float
        # width, height: float
        # incidence_matrix: numpy array
        # time_step: int
        # self.observation_space = spaces.Tuple(
            # (
                # spaces.Box(
                #     low=0.0,
                #     high=max(self.canvas.width, self.canvas.height),
                #     shape=(self.canvas.num_boxes, 2),
                #     dtype=np.float32,
                # ),
                # spaces.Box(
                #     low=0.0,
                #     high=self.canvas.width,
                #     shape=(self.canvas.num_boxes, 2),
                #     dtype=np.float32,
                # ),
                # spaces.MultiBinary(
                #     self.canvas.num_boxes * self.canvas.num_nets
                # ),  # incidence matrix
                # spaces.Discrete(self.time_limit),
            # )
        # )

        # use a dict for observation
        self.observation_space = spaces.Dict(
            {   "box_cxy": spaces.Box(
                    low=0.0,
                    high=max(self.canvas.width, self.canvas.height),
                    shape=(self.canvas.num_boxes, 2),
                    dtype=np.int32,
                ),
                "width_height": spaces.Box(
                    low=0.0,
                    high=max(self.canvas.width, self.canvas.height),
                    shape=(self.canvas.num_boxes, 2),
                    dtype=np.int32,
                ),
                "incidence_matrix": spaces.MultiBinary(
                    [self.canvas.num_boxes, self.canvas.num_nets]
                ),  # incidence matrix
                "time_step": spaces.Discrete(self.time_limit),
            }
        )

    def step(self, action):
        """
        action: (box_name, cx, cy)
        """
        # print(action)
        box_name_i, cx, cy = action
        box_name=f"box_{int(box_name_i)}"
        self.canvas.set_box_cxy(box_name, cx, cy)
        # self.canvas.move_all_boxes() # move all the boxes is not necessary if directly set the center of the box
        self.time_step += 1
        # compute the reward
        reward = -self.canvas.get_total_wirelength_from_cxy()
        # check if the episode is done
        done = False
        if self.time_step >= self.time_limit:
            done = True
        # compute the info
        info = {}
        return self._get_obs(), reward, done, False, info
    
    def reset(self, seed=None):
        """
        reset the environment
        """
        super().reset(seed=seed)
        self.time_step = 0
        with open(f"./testcase/pcb_canvas_09.pkl", "rb") as f:
            self.canvas = pickle.load(f)
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        """
        return the observation
        """
        # put width and height into one array
        width_height = np.zeros((self.canvas.num_boxes, 2))
        for i, box in enumerate(self.canvas.iter_box()):
            width_height[i, 0] = box.width * 1.0
            width_height[i, 1] = box.height * 1.0
        # return (
            # self.canvas.get_cxy().astype(np.float32),
            # width_height.astype(np.float32),
            # self.canvas.get_incidence_matrix().astype(np.float32),
            # self.time_step,
        # )

        return {
            "box_cxy": self.canvas.get_cxy().astype(np.int32),
            "width_height": width_height.astype(np.int32),
            "incidence_matrix": self.canvas.get_incidence_matrix().astype(np.int8),
            "time_step": self.time_step,
        }
    
    def _get_info(self):
        """
        return the info
        """
        return {
            "canvas": self.canvas,
        }
    
    def render(self, mode="human"):
        """
        render the environment
        """
        if mode == "human":
            self.canvas.plot()
        elif mode == "graph":
            self.canvas.plot(savefig=True, filename="pcb_env.png")
        else:
            super(PCB_Environment, self).render(mode=mode)


        
