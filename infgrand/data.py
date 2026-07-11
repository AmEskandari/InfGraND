"""Dataset loading for the six datasets in InfGraND Table 1."""
import dgl
import torch

from .utils import set_seed

_DGL_LOADERS = {
    "cora": dgl.data.CoraGraphDataset,
    "citeseer": dgl.data.CiteseerGraphDataset,
    "pubmed": dgl.data.PubmedGraphDataset,
    "coauthor-cs": dgl.data.CoauthorCSDataset,
    "coauthor-phy": dgl.data.CoauthorPhysicsDataset,
    "amazon-photo": dgl.data.AmazonCoBuyPhotoDataset,
}


def load_dataset(name: str, seed: int = 2022):
    """Return (graph, labels, idx_train, idx_val, idx_test).

    For Cora/Citeseer/Pubmed we use the standard public splits. For Coauthor
    and Amazon-Photo we sample 20 nodes/class for train, 500 for val, 1000 for
    test, seeded so the split is reproducible across runs.
    """
    if name not in _DGL_LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported: {sorted(_DGL_LOADERS)}"
        )
    graph = _DGL_LOADERS[name](verbose=False)[0]
    labels = graph.ndata["label"]
    g = dgl.add_self_loop(dgl.remove_self_loop(graph))

    if name in {"cora", "citeseer", "pubmed"}:
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
    else:
        set_seed(seed)
        n = labels.shape[0]
        n_class = int(labels.max().item() + 1)
        nrange = torch.arange(n)
        train_mask = torch.zeros(n, dtype=torch.bool)
        for y in range(n_class):
            cls = nrange[labels == y]
            train_mask[cls[torch.randperm(cls.shape[0])[:20]]] = True
        val_mask = ~train_mask
        not_train = nrange[val_mask]
        val_mask[not_train[torch.randperm(not_train.shape[0])[500:]]] = False
        test_mask = ~(train_mask | val_mask)
        not_tv = nrange[test_mask]
        test_mask[not_tv[torch.randperm(not_tv.shape[0])[1000:]]] = False

    return (
        g,
        labels,
        train_mask.nonzero(as_tuple=False).squeeze(-1),
        val_mask.nonzero(as_tuple=False).squeeze(-1),
        test_mask.nonzero(as_tuple=False).squeeze(-1),
    )
