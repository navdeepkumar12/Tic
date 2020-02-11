import numpy as np
from test import grid
for i in range(3**9):
    temp = np.base_repr(i,base=3)
    temp = "".join(['0'*(9-len(temp)),temp])
    print(i, temp)
    grid(temp)
         