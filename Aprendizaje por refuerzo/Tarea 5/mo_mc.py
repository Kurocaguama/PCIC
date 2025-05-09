import math
import numpy as np
from typing import Optional
from gymnasium import spaces
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.utils import EzPickle

class MO_MC(MountainCarEnv, EzPickle):
    """
    Vamos a modificar el caso singular del entorno MountainCar
    de manera similar a lo que se hace en MO-Gymnasium, pero
    considerando diferentes recompensas.

    La recompensa original es -1 por cada timestep que el coche
    no llegue a la meta, para nuestro caso agregaremos unas 
    recompensas nuevas para tener el caso multi-objetivo. Como
    tal las recompensas serán de velocidad máxima del coche
    """
    def __init__(
            self,
            render_mode,
            goal_velocity = 0,
    ):
        super().__init__(render_mode, goal_velocity)
        EzPickle.__init__(self, render_mode, goal_velocity)
        self.reward_dim = 2

        # Dimensiones de la recompensa
        low = np.array([-1]*self.reward_dim)
        high = np.zeros(self.reward_dim)
        low[-1] = 0.0
        high[-1] = 1.1
        high[0] = -1

        # Espacio de recompensa
        self.reward_space = spaces.Box(low=low, high=high, shape=(self.reward_dim,), dtype = np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        pos, vel = self.state
        vel += (action-1) * self.force + math.cos(3*pos) * (-self.gravity)
        vel = np.clip(vel, -self.max_speed, self.max_speed)
        pos += vel
        pos = np.clip(pos, self.min_position, self.max_position)
        if pos == self.min_position and vel < 0 :
            vel = 0

        term = bool(pos >= self.goal_position and vel >= self.goal_velocity)
        reward = np.zeros(self.reward_dim, dtype = np.float32)
        reward[0] = 0.0 if term else -1.0 # Fin del juego
        reward[-1] = 15* abs(vel) # Premio en función de la velocidad
        #print(reward)

        self.state = (pos, vel)
        if self.render_mode == 'human':
            self.render()
        return np.array(self.state, dtype=np.float32), reward, term, False, {}


env = MO_MC('human')
env.reset()
env.step(1)