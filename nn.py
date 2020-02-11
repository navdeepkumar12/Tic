import numpy as np 

class layer:
    def __init__(self):
        self.w = []
        self.b = []
        self.input =[]
        self.output = []
        self.gradient = []


class linear(layer):
    def __init__(self):
        super().__init__(layer)
