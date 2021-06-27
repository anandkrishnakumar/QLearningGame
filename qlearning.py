import numpy as np
import time

def Qtrain(Game, agent, episodes=1000):
    """Q-learning training"""
    a = agent
    times_taken = np.zeros(episodes)
    print("Training starting")
    for n in range(episodes):
        start_time = time.time()
        g = Game()
        # print("EPISODE", n+1)
        while not g.success:
            state = 1.0*g.get_state()
            action = a.action(state)
            reward = g.play(action)
            # print(g.success)
            # print("reward: ", reward)
            # print(state)
            # print(g.get_state())
            a.train(state, action, reward, g.get_state(), g.success)
        end_time = time.time()
        times_taken[n] = end_time - start_time
    print("Training complete ({} episodes)".format(episodes))
    return times_taken