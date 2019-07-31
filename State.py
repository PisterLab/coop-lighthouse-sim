
from py3dmath import Vec3, Rotation

class State:
    def __init__(self, pos, vel, att):
        self._pos = pos
        self._vel = vel
        self._att = att

    def vectorize(self):
        return [self._pos[0], self._pos[1], self._vel[0], self._vel[1], self._att]
