import torch

def load_data(dataset, batch_size, shuffle=True, seed=None):
    if batch_size == -1:
        batch_size = len(dataset)
    data = dataset.data
    targets = dataset.targets
    if seed is not None:
        torch.manual_seed(seed)
    if shuffle:
        idxs = torch.randperm(len(dataset))
    else:
        idxs = torch.arange(len(dataset))

    for i in range(len(dataset) // batch_size):
        lhs = i * batch_size
        rhs = (i + 1) * batch_size
        idx = idxs[lhs:rhs]
        yield (data[idx], targets[idx])


class Loader:
    def __init__(self, dataset, batch_size=-1, shuffle=True, device=None) -> None:
        if batch_size == -1:
            batch_size = len(dataset)
        self.loader = load_data(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        if device is not None:
            self.loader = [(d.to(device), t.to(device)) for d,t  in self.loader]
        self.length = len(dataset) // batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.loader)


def accuracy(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).float().mean() * 100
