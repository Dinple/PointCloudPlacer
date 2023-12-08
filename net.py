import numpy as np

class Net:
    def __init__(self, net_name, source_box_name, sink_box_names):
        """
        boxes only perseve the name of the box, not the box itself
        """
        self.net_name = net_name
        # source box at index 0, sink boxes at index 1...n
        self.box_names = []
        self.source_box_name = source_box_name

        if source_box_name is not None:
            self.box_names.append(source_box_name)
            for box_name in sink_box_names:
                self.box_names.append(box_name)

        # net has intrinsic properties
        # rotation angle
        self.dr = 0

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
