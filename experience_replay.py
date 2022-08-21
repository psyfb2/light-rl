import numpy as np
from collections import namedtuple


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "terminal")
)


class ReplayBuffer:
    """Replay buffer to sample experience/ transition tuples from

    :attr capacity (int): total capacity of the replay buffer
    :attr memory (Transition):
        Each component of the transition tuple is represented by a zero-initialised np.ndarray of
        floats with dimensionality (total buffer capacity, component dimensionality)
    :attr writes (int): number of experiences/ transitions already added to the buffer
    """

    def __init__(self, capacity: int):
        """Constructor for a ReplayBuffer initialising an empty buffer (without memory
        
        :param capacity (int): total capacity of the replay buffer
        """
        self.capacity = int(capacity)
        self.memory = None
        self.writes = 0

    def init_memory(self, transition: Transition):
        """Initialises the memory with zero-entries

        :param transition (Transition): transition(s) to take the dimensionalities from
        """
        for t in transition:
            assert t.ndim == 1  # sanity check

        self.memory = Transition(
            *[np.zeros([self.capacity, t.size], dtype=t.dtype) for t in transition]
        )

    def push(self, *args):
        """Adds transitions to the memory

        Note:
            overwrites first transitions stored once the capacity limit is reached

        :param *args: arguments to create transition from
        """
        if not self.memory:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes = self.writes + 1

    def sample(self, batch_size: int) -> Transition:
        """Samples batch of experiences from the replay buffer

        :param batch_size (int): size of the batch to be sampled and returned
        :return (Transition): batch of experiences of given batch size
        """
        samples = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[np.take(d, samples, axis=0) for d in self.memory]
        )
        return batch

    def __len__(self):
        """Gives the length of the buffer
        """
        return min(self.writes, self.capacity)
