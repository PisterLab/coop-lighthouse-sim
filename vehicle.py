import numpy as np
import State

class Vehicle:
    def __init__(self, init_state=State()):
        self.state_truth = [init_state]
