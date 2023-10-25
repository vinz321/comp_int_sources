import numpy as np
from enum import Enum
class Directions(Enum):
    LEFT=[1,0]
    UP=[0,1]
    RIGHT=[-1,0]
    DOWN=[0,-1]

class Walls(Enum):
    NONE=0
    VERTICAL=2
    HORIZONTAL=3