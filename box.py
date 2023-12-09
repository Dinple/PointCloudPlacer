import numpy as np

class Box:
    def __init__(self, box_name, llx, lly, width, height, net_name_list):
        self.box_name = box_name
        self.width = width
        self.height = height
        # self.net_name = net_name
        self.net_name_list = net_name_list
        # lower left corner
        self.llx = llx
        self.lly = lly
        # center
        self.cx = llx + width / 2
        self.cy = lly + height / 2
        # upper right corner
        self.urx = llx + width
        self.ury = lly + height
        # rotation
        self.rotation = 0
        # update motion
        self.dx = 0
        self.dy = 0
        self.dr = 0  # between 0 to 360
        # motion decay
        self.decay = 0.9
        # print("[INFO] Box %s created" % self.box_name)

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
        
    def add_net_name(self, net_name):
        """
        add a net name to the net name list
        """
        self.net_name_list.append(net_name)

    def self_rotate(self):
        """
        based on the motion, rotate the box by the angle of the motion
        """
        # compute rotation speed based on the motion and the size of the box
        rotation_speed = np.sqrt(self.dx**2 + self.dy**2) / (
            self.width + self.height
        )

        # compute the angle of the motion
        if self.dx == 0:
            if self.dy > 0:
                self.dr = 90
            elif self.dy < 0:
                self.dr = 270
            else:
                self.dr = 0
        else:
            self.dr = np.arctan(self.dy / self.dx) / np.pi * 180

        # set the rotation
        self.rotation += self.dr * rotation_speed

        # normalize the rotation
        if self.rotation > 360:
            self.rotation -= 360
        elif self.rotation < 0:
            self.rotation += 360
        # print("[INFO] Box %s: rotation: %f" % (self.box_name, self.rotation))

        