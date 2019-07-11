
class Measurement:
    def __init__(self):
        self.Pm = [np.diag([self.sig_x0, self.sig_y0, self.sig_th0, self.sig_vx0, self.sig_vy0])]
