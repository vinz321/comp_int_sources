import os
import pathlib
import numpy as np
from time import sleep
from configparser import ConfigParser
import snake

PATH=pathlib.Path(__file__).parent.resolve()
print(PATH)
os.chdir(PATH)
parser=ConfigParser()
parser.read('./config.ini')
SCREEN_SIZE=np.array([int(parser.get('Default','SCREEN_HEIGHT')), int(parser.get('Default','SCREEN_WIDTH'))])

table=np.zeros(SCREEN_SIZE, dtype='uint')
table[8, 12]=2
table[0,:]=1
table[:,0]=1
table[SCREEN_SIZE[0]-1,:]=1
table[:,SCREEN_SIZE[1]-1]=1
player=snake.Snake(SCREEN_SIZE, table)


def print_screen(table:np.ndarray):
    for i in table:
        print(i)
while(True):
    os.system('cls')
    
    print_screen(table)
    if not player.step(table):
        break
    sleep(.5)
