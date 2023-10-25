import numpy as np
from util_values import Directions, Walls

class Player:
    position=(0,0)
    def __init__(self, starting_point, table, wallTable):
        self.position=np.array(starting_point, dtype='int')
        self.table=table
        self.wallTable=wallTable
        print(wallTable.shape)

    def get_edge_values(self, i,j, direction):
        if(direction==Directions.UP):
            return [self.wallTable[i,j+1], self.wallTable[i+1,j+1]]
        elif(direction==Directions.DOWN):
            return [self.wallTable[i,j], self.wallTable[i+1,j]]
        elif(direction==Directions.LEFT):
            return [self.wallTable[i,j], self.wallTable[i,j+1]]
        elif(direction==Directions.LEFT):
            return [self.wallTable[i+1,j], self.wallTable[i+1,j+1]]
        
        
    def can_move(self, direction):
        u,v = self.get_edge_values(self.position[0],self.position[1],direction)

        if direction==Directions.UP or direction==Directions.DOWN:
            return u!=Walls.HORIZONTAL or v!=Walls.HORIZONTAL
        if direction==Directions.LEFT or direction==Directions.RIGHT:
            return u!=Walls.VERTICAL or v!=Walls.VERTICAL    

    def move(self, direction):
        if(self.can_move(direction)):
            self.table[self.position]=0
            self.position=(self.position[0]+direction.value[0], self.position[1]+direction.value[1])

            if self.position[0]<0 or self.position[0]>=self.table.shape[0]:
                self.position=(self.position[0]-direction.value[0], self.position[1]-direction.value[1])
                return False
            if self.position[1]<0 or self.position[1]>=self.table.shape[1]:
                self.position=(self.position[0]-direction.value[0], self.position[1]-direction.value[1])
                return False 
            self.table[self.position]=1
            print(self.position)    
            return True
        else:
            return False
