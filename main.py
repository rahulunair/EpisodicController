from mfec import EPCAgent
from mfec import HPS

class Environment(object):
    def __init__(self, env):
        self.env = globals()[env]()
        self.state_size = self.env.observation_space.shape[0]
        self.actions_size = self.env.action_space.n


if __name__ == "__main__":
    env = Environment("TempEnv")
    agent = EPCAgent(env.actions_size, env.state_size)
    env = env.env  # the wrapped environment from Environment
    state_size = env.observation_space.shape[0]
    actions_size = env.action_space.n
    agent = EPCAgent(state_size, actions_size)
    logger = petri.Petri("debug")
    episodes = 800
    max_steps = 250
    reward = 0
    done = False
    additive_rewards = 0

    for e in range(episodes):
        print("=" * 45)
        print("Episode: ", e)
        steps = 0
        state = env.reset()
        rewards = 0
        for s in range(max_steps):
            action = agent.play(state, reward, done)
            state, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
        additive_rewards = additive_rewards * 0.95 + rewards * 0.05
        if e % 40 == 0:
            print("Reward till episode {} is:{} ".format(e, additive_rewards))
