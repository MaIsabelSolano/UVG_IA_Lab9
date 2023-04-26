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
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time 

nIterations = 100

env = gym.make("FrozenLake-v1", render_mode = "human", desc=generate_random_map(size=4))


for n in range (nIterations):

    print(f"Iteration no. {n+1}")

    env.reset()
    env.render()

    numberOfSteps = 100

    for i in range(numberOfSteps):
        randomAction = env.action_space.sample()
        returnValue = env.step(randomAction)
        # print(returnValue)
        env.render()

        print(f"step no. {i+1}")
        time.sleep(0.20)

        if returnValue[1] > 0:
            print("Win\n")
            env = gym.make("FrozenLake-v1", render_mode = "human", desc=generate_random_map(size=4))
            break

        if returnValue[2]:
            print("Game Over\n")
            break


env.close()