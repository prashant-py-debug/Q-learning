import gym
import tensorflow as tf
import numpy as np 

ngames = 10
load_model = "models/dqn_200_model.h5"

agent = tf.keras.models.load_model(load_model)

env = gym.make("LunarLander-v2")

for _ in range(ngames):
    done = False
    obeservation = env.reset()
    while not done:
        action = np.argmax(agent.predict((obeservation).reshape(1,8)))
        # action = env.action_space.sample()
        new_state , reward, done, _ = env.step(action)
        env.render()

        obeservation = new_state

env.close()

    