import torch
import numpy as np


def scatter(src: torch.Tensor, index: torch.Tensor, device: str, reduce_op: str) -> torch.Tensor:
    """
     For details see:
    https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/min.html
    and
    https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/max.html

    Args:
        src: the actual values that are to be minimized or maximized
        index: the indices to be minimized or maximized
        device: provide the device on which src and index args reside
        reduce_op: type of reduce operation to perform options: min | max

    returns: A tensor having min or max values grouped by index arg at the indices
    specified by position var

    """
    # src and index tensor must be of same size
    src = src.cpu().numpy()
    index = index.cpu().numpy()
    assert src.shape == index.shape
    # the array where min/max outputs are placed and +1 to account for the offset. empty slots are zeros
    output = np.zeros(index.max() + 1, dtype=src.dtype)
    # creates a sorted array of indices
    idx_sort = np.argsort(index)
    # indexing the original array with sorted indices
    sorted_records_array = index[idx_sort]
    # get the positions where min/max will be stored and the indices where those positions were found
    position, idx_start = np.unique(sorted_records_array, return_index=True)
    # getting variable chunk sizes i.e. [array([0, 1, 3]), array([2]), array([4, 5]), array([6, 7])]
    # represents that elements at these indices will be minimized or maximized later. zero omitted
    chunks = np.split(idx_sort, idx_start[1:])
    # reduce min operation
    if reduce_op == 'min':
        for pos, chunk in zip(position, chunks):
            output[pos] = min(src[chunk])
        return torch.from_numpy(output).to(device)
    # reduce max operation
    elif reduce_op == 'max':
        for pos, chunk in zip(position, chunks):
            output[pos] = max(src[chunk])
        return torch.from_numpy(output).to(device)
    # non supported operations
    else:
        raise ValueError(f"Invalid reduce operation type: {reduce_op}")
