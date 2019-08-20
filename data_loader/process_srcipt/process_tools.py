import torch


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    # print(src)
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge_indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.unsqueeze(0).expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


def adj_dict_2_torch(edge_dict):
    keys = list(edge_dict.keys())
    # print(keys)
    node_size = max(keys)
    torch_adj = torch.zeros([node_size + 1, node_size + 1], dtype=torch.float32)
    # print(torch_adj)
    for node_index,  ad_nodes in edge_dict.items():
        ad_nodes = torch.tensor(ad_nodes, dtype=torch.long)
        torch_adj[node_index, ad_nodes] = 1
        # print(index, ad_nodes)
    # print(torch_adj)
    return torch_adj


def normalize(data):
    # node * fz
    data = data / data.sum(1, keepdim=True).clamp(min=1)
    return data

if __name__ == '__main__':
    test_dict = {0:[1], 2:[1,2]}
    adj_dict_2_torch(test_dict)