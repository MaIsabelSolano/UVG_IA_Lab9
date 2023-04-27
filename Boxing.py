"""
Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la computación
Inteligencia Artificial 

Integrantes: 
- Christopher García
- Alejandro Gómez
- Ma. Isabel Solano 

Referencia: https://aleksandarhaber.com/installation-and-getting-started-with-openai-gym-and-frozen-lake-environment-reinforcement-learning-tutorial/
"""

import gymnasium as gym
import time 
# import atari_py

nIterations = 300

env = gym.make("ALE/Boxing-v5", render_mode = "human")
env.metadata['render_fps'] = 60

env.reset()
env.render()

randomAction = env.action_space.sample()
returnValue = env.step

