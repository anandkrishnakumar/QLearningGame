def test(Game, agent, init=None, weird=False):
    """Testing an agent"""
    print("\n\nTest")
    a = agent
    old_eps = a.epsilon
    a.epsilon = 0.01 # not much exploration during testing
    g = Game()
    if init is not None:
        g.set_state(init)
    while not g.success:
        state = 1.0*g.get_state()
        action = a.action(state, weird)
        reward = g.play(action)
        print(state)
        if g.success:
            print("Success")
        print("reward: ", reward)
    print(g.get_state())
    a.epsilon = old_eps