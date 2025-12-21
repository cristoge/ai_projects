import pyamaze as maze
from queue import PriorityQueue
import time

ROWS = 20
COLS = 20


def distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)


# def aStar(m):
#    return forwardPath

m = maze.maze(ROWS, COLS)
m.CreateMaze()
pre_Astar = time.time()
path = aStar(m)
post_Astar = time.time()
print(post_Astar - pre_Astar)
a = maze.agent(m, footprints=True)
m.tracePath({a: path}, delay=5)
m.run()
