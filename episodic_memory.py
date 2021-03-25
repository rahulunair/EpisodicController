class Memory(object):
    """Episodic Memory"""

    def __init__(self, size, state_size):
        self.states = np.zeros((size, state_size))
        self.actions = {}
        self.rewards = {}
        self.values = {}
        self.index = 0
        self.end = 0
        self.episode_pointer = 0
        self.size = size

    def k_nearest(self, obs, neigh):
        """return K nearest neighbours from memory"""
        # TODO(unrahul): replace this with a KD-tree or BK-tree
        curr_states = self.states[:self.end]
        l2 = np.sum((curr_states - obs) ** 2, axis=1)  # Order of N
        indices = np.argsort(l2)
        return indices[: min(neigh, len(indices))]
