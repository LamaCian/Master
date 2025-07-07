import random
from collections import defaultdict
from torch.utils.data import Sampler

class UniqueSampler(Sampler):
    def __init__(self, caption_ids, batch_size, drop_last=False, seed=None):
        super().__init__()
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.seed       = seed

        self.group_to_indices = defaultdict(list)
        for idx, cid in enumerate(caption_ids):
            self.group_to_indices[cid].append(idx)

    def __iter__(self):
        # Optional reproducible shuffling
        rnd = random.Random(self.seed)

        # Make fresh copies each epoch
        local_groups  = list(self.group_to_indices)
        local_indices = {cid: ids.copy() for cid, ids in self.group_to_indices.items()}

        rnd.shuffle(local_groups)
        for lst in local_indices.values():
            rnd.shuffle(lst)

        batch = []
        for cid in local_groups:
            if not local_indices[cid]:
                continue
            batch.append(local_indices[cid].pop())
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        available = sum(1 for ids in self.group_to_indices.values() if ids)
        if self.drop_last:
            return available // self.batch_size
        return (available + self.batch_size - 1) // self.batch_size
