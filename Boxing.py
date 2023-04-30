"""
Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la computación
Inteligencia Artificial 

Integrantes: 
- Christopher García
- Alejandro Gómez
- Ma. Isabel Solano 
- Roberto Vallecillos

Referencia: https://aleksandarhaber.com/installation-and-getting-started-with-openai-gym-and-frozen-lake-environment-reinforcement-learning-tutorial/
"""

import gym
import ale_py
import shimmy
import numpy as np
import atari_py

env = gym.make("Boxing-v3")
env.metadata["render_fps"] = 60

env.reset()
env.render()
randomAction = env.action_space.sample()
returnValue = env.step

action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]
q_table = np.zeros((state_space_size, action_space_size))

learning_rate = 0.1
discount_factor = 0.99
rateExploroacion = 1
maxRateExploroacion = 1
minRateExploroacion = 0.01
decay = 0.001

nIterations = 300
for i in range(nIterations):
    obs = env.reset()
    done = False
    rateExploroacion = minRateExploroacion + (
        maxRateExploroacion - minRateExploroacion
    ) * np.exp(-decay * i)
    while not done:
        if np.random.uniform(0, 1) < rateExploroacion:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[obs])

        new_obs, reward, done, info = env.step(action)

        maxqFutura = np.max(q_table[new_obs])
        qActual = q_table[obs, action]
        qNueva = (1 - learning_rate) * qActual + learning_rate * (
            reward + discount_factor * maxqFutura
        )
        q_table[obs, action] = qNueva
        obs = new_obs
    print("Iteración: %d, Punteo: %d" % (i + 1, info["score"]))

obs = env.reset()
done = False
while not done:
    action = np.argmax(q_table[obs])
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
