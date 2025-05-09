import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces, utils
from typing import Optional

# 0 = Free spot
# 1 = Blocked spot
# 2 = Coin
# 3 = End 
maze = np.array([
    [0,0,1,1,2],
    [2,0,2,1,0],
    [0,0,1,0,0],
    [1,0,0,0,1],
    [2,0,1,0,3],
])

# Acciones posibles: 0,1,2,3
# 0 -> Arriba
# 1 -> Derecha
# 2 -> Abajo
# 3 -> Izquierda

class Maze(Env):
    """
    Esto es un entorno sencillo que corresponde a un laberinto (maze) cuadricular de 5x5.
    """
    def __init__(
            self,
            maze
      ):
        self.maze = maze
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(5*5) # Tamaño del laberinto
        self.x = 0
        self.y = 0
        self.current_state = self.maze[self.x][self.y]
        self.seed = 42

    def print_state(self):
        print("({},{})".format(self.x, self.y))

    def move(self, action):
        min_lim = 0
        max_lim = 4
        aux_x = self.x
        aux_y = self.y
        if action == 0 and self.x - 1 >= min_lim:
            aux_x = self.x - 1
        elif action == 1 and self.y + 1 <= max_lim:
            aux_y = self.y + 1
        elif action == 2 and self.x + 1 <= max_lim:
            aux_x = self.x + 1
        elif action == 3 and self.y -1 >= min_lim:
            aux_y = self.y - 1
        if self.maze[aux_x][aux_y] != 1:
            self.x = aux_x
            self.y = aux_y
            self.current_state = self.maze[self.x][self.y]
        
        if self.current_state == 2:
            self.maze[self.x][self.y] = 0

    def get_rewards(self):
        if self.current_state == 0:
            return -1
        elif self.current_state == 2:
            return 1
        elif self.current_state == 3:
            return 10
        
    def step(self, action):
        self.move(action)
        obs = self.maze
        state = self.current_state
        reward = self.get_rewards()
        term = bool(self.current_state == 3)
        return obs, state, reward, term
    
    def reset(
            self,
            *,
            seed: Optional[int],
    ):
        super().reset(seed=self.seed)
        self.x = 0
        self.y = 0
        self.current_state = self.maze[self.x][self.y]
        self.maze = maze



aux = Maze(maze)
aux.reset()
print(aux.step(2))
