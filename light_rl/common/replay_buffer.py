import numpy as np
from collections import namedtuple


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "terminal")
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        """ Replay buffer to sample experience/ transition tuples from.
        
        Args:
            capacity (int): number of samples which this replay buffer can store
        """
        self.capacity = int(capacity)
        self.memory = None
        self.writes = 0

    def init_memory(self, transition: Transition):
        """ Initialise memory with zero-entries

        Args:
            transition (Transition): transition to take the dimensionalities from
        """
        for t in transition:
            if t.ndim != 1:
                raise ValueError(f"transition entry {t} is {t.ndim} dimensional, must be one dimensional")

        self.memory = Transition(
            *[np.zeros([self.capacity, t.size], dtype=t.dtype) for t in transition]
        )

    def push(self, *args):
        """ Adds single transitions to the buffer
        
        Args:
            *args: arguments to create single transition from
        """
        if not self.memory:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes = self.writes + 1

    def sample(self, batch_size: int) -> Transition:
        """ Sample batch of experiences from the replay buffer

        Args:
            batch_size (int): size of the batch to be sampled and returned
        Returns:
            (Transition): batch of experiences of given batch size
        """
        samples = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[np.take(d, samples, axis=0) for d in self.memory]
        )
        return batch

    def __len__(self) -> int:
        """ Get the length of the buffer
        Returns:
            (int): length of buffer
        """
        return min(self.writes, self.capacity)
