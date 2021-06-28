"""Tests

This file contains the following functions:
    
    * test - to test an agent
"""

def test(Game, agent, init=None, weird=False):
    """Tests an agent."""
    print("\n\nTest")
    a = agent
    old_eps = a.epsilon
    a.epsilon = 0.01 # not much exploration during testing
    g = Game()
    if init is not None:
        g.set_state(init)
    moves = 0 # to prevent infinite loops
    while (not g.success) and moves < 50:
        state = 1.0*g.get_state()
        action = a.action(state, weird)
        reward = g.play(action)
        print(state)
        if g.success:
            print("Success")
        print("reward: ", reward)
        moves += 1
    print(g.get_state())
    a.epsilon = old_eps