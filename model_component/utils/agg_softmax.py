
from torch_scatter import scatter_max, scatter_add
# from .num_nodes import maybe_num_nodes


def softmax(src, index, num_nodes=None):
    r"""Sparse softmax of all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    # print('src = ',src.size(), ' index = ',index.size())

    if num_nodes is None:
        # num_nodes = maybe_num_nodes(index, num_nodes)
        num_nodes = index.max()+1

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    # print('src - scatter_max = ',out.size(),out)
    out = out.exp()
    # print('exp = ',out.size(),out, 'index size = ',index.size())
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    # print('out = ',out.size(),out)
    return out