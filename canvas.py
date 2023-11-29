import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding
import re
import os
import imageio


class PCBCanvas:
    def __init__(self, width: float, height: float, boxes: dict, nets: dict):
        self.width = width
        self.height = height
        # entities
        self.boxes = boxes
        self.nets = nets
        self.num_boxes = len(boxes)
        self.num_nets = len(nets)

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
        return (box.cx, box.cy)

    def get_cxy(self):
        """
        return a list of (cx, cy) of the boxes in the canvas
        """
        cxy_list = []
        for box in self.iter_box():
            cxy_list.append((box.cx, box.cy))
        return np.array(cxy_list)

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

    def move_all_boxes(self):
        """
        move all the boxes
        """
        for box in self.iter_box():
            self.move_box(box.box_name, if_decay=True)
            # TODO
            # /*--- add your method here ---*/
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
                    unit_vector = (
                        cxy[ind[i, 1]] - cxy[i]
                    ) / np.linalg.norm(cxy[ind[i, 1]] - cxy[i])
                    # print(unit_vector)
                    # compute the motion
                    dx = -unit_vector[0] * scale
                    dy = -unit_vector[1] * scale
                    # print(dx, dy)
                    # set the motion
                    box.set_motion(box.dx + dx, box.dy + dy)


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
                    np.array(source_box_cxy),
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
                total_wirelength += np.linalg.norm(
                    np.array(source_box_cxy) - np.array(box_cxy)
                )

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
            ax.add_patch(
                plt.Rectangle(
                    (box.llx, box.lly),
                    box.width,
                    box.height,
                    facecolor="none",
                    edgecolor="black",
                )
            )
        # plot the nets
        for net_box_list in self.iter_net_box():
            x = []
            y = []
            for box in net_box_list:
                x.append(box.cx)
                y.append(box.cy)
            ax.plot(x, y, "r-")
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


class Box:
    def __init__(self, box_name, llx, lly, width, height, net_name):
        self.box_name = box_name
        self.width = width
        self.height = height
        self.net_name = net_name
        # lower left corner
        self.llx = llx
        self.lly = lly
        # center
        self.cx = llx + width / 2
        self.cy = lly + height / 2
        # upper right corner
        self.urx = llx + width
        self.ury = lly + height
        # update motion
        self.dx = 0
        self.dy = 0
        # motion decay
        self.decay = 0.9
        print("[INFO] Box %s created" % self.box_name)

    # Coordinate update
    def set_ll_xy(self, llx, lly):
        """
        set the lower left corner of the box
        """
        self.llx = llx
        self.lly = lly
        self.update_center_xy()
        self.update_ur_xy()

    def set_center_xy(self, cx, cy):
        """
        set the center of the box
        """
        self.cx = cx
        self.cy = cy
        self.update_ll_xy()
        self.update_ur_xy()

    def set_ur_xy(self, urx, ury):
        """
        set the upper right corner of the box
        """
        self.urx = urx
        self.ury = ury
        self.update_ll_xy()
        self.update_center_xy()

    def update_ll_xy(self):
        """
        update the lower left corner of the box
        """
        self.llx = self.cx - self.width / 2
        self.lly = self.cy - self.height / 2

    def update_center_xy(self):
        """
        update the center of the box
        """
        self.cx = self.llx + self.width / 2
        self.cy = self.lly + self.height / 2

    def update_ur_xy(self):
        """
        update the upper right corner of the box
        """
        self.urx = self.llx + self.width
        self.ury = self.lly + self.height

    def set_motion(self, dx, dy):
        """
        set the motion
        """
        self.dx = dx
        self.dy = dy

    def decay_motion(self):
        """
        decay the motion
        """
        self.dx *= self.decay
        self.dy *= self.decay


class Net:
    def __init__(self, net_name, source_box_name, sink_box_names):
        """
        boxes only perseve the name of the box, not the box itself
        """
        self.net_name = net_name
        # source box at index 0, sink boxes at index 1...n
        self.box_names = []
        if source_box_name is not None:
            self.box_names.append(source_box_name)
            for box_name in sink_box_names:
                self.box_names.append(box_name)

    def set_source_box_name(self, box_name):
        """
        set the source box name
        """
        assert len(self.box_names) == 0
        self.box_names.append(box_name)

    def add_sink_box_name(self, box_name):
        """
        add a sink box name
        """
        self.box_names.append(box_name)

    def iter_box_name(self):
        """
        iterate through the box names
        """
        for box_name in self.box_names:
            yield box_name

    def get_source_box_name(self):
        """
        return the source box name
        """
        return self.box_names[0]

    def get_sink_box_name(self, i):
        """
        return the sink box name at index i
        """
        assert i > 0
        return self.box_names[i]

    def get_sink_box_names(self):
        """
        return the list of sink box names
        """
        return self.box_names[1:]

    @staticmethod
    def get_net_spring_force(self, point_A, point_B, vel_A, vel_B, k, l0, kd):
        """
        compute the net spring force between point A and point B
        """
        # compute the distance between point A and point B
        distance = np.linalg.norm(point_A - point_B)
        # compute the unit vector from point A to point B
        unit_vector = (point_B - point_A) / distance
        # compute the spring force
        F_spring = k * (distance - l0) * unit_vector
        # compute the damping force
        F_damping = kd * (np.dot(vel_A - vel_B, unit_vector)) * unit_vector
        # compute the net spring force
        F_net = F_spring - F_damping
        return F_net
