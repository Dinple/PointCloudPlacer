import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding
import os


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
        incidence_matrix = self.get_incidence_matrix()
        # Compute the difference in coordinates for each edge
        edge_coords = np.dot(incidence_matrix.T, cxy)
        # Compute the wirelength
        edge_lengths = np.linalg.norm(edge_coords, axis=1, ord=2)
        total_wirelength = np.sum(edge_lengths)

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


class Box:
    def __init__(self, box_name, llx, lly, width, height, net_name):
        self.box_name = box_name
        self.width = width
        self.height = height
        # lower left corner
        self.llx = llx
        self.lly = lly
        # center
        self.cx = llx + width / 2
        self.cy = lly + height / 2
        # upper right corner
        self.urx = llx + width
        self.ury = lly + height

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
