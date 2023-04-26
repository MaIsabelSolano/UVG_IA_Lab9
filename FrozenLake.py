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

env = gym.make("FrozenLake-v1", render_mode = "human")

env.reset()

env.render()

numberOfSteps = 30

for i in range(numberOfSteps):
    randomAction = env.action_space.sample()
    returnValue = env.step(randomAction)
    env.render()

    print(f"iteration {i}")
    time.sleep(1)

    if returnValue[2]:
        break

env.close()