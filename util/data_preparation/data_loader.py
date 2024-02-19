import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset,IterableDataset

class GeneratorDataset(IterableDataset):
    def __init__(self, gen_func,batch_size):
        super().__init__()
        self.gen_func = gen_func(batch_size)

    def __iter__(self):
        return self.gen_func

class GeneratorDataLoader(DataLoader):
    def __init__(self, gen_func, batch_size=1,shuffle=False, num_workers=0):
        dataset = GeneratorDataset(gen_func,batch_size)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
