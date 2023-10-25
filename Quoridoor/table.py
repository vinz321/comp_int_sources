import numpy as np
from util_values import Directions, Walls
from player import Player

p1=(1,1)

pawn_table=np.zeros([9,9],dtype='uint')

# How to codify wall_table: 
# Set the value of the point
# The point corresponds to the intersection between cells
# the point (i,j) is the intersection between (i-1,j), (i,j), (i,j-1), (i-1,j-1) 

wall_table=np.zeros([10,10], dtype='uint')

direction=Directions.UP

p=Player(p1, pawn_table, wall_table)

test=True
while(test):
    test=p.move(direction)

