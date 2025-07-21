from collections import defaultdict
from heapq import merge

import numpy as np
import pandas as pd
import torch
from data.embeddings import decode_batch_of_embeddings
from tabulate import tabulate
from torch.nn.utils.rnn import pad_sequence


def merge_dicts_sorted(
    d1: dict[str, list[int]], d2: dict[str, list[int]]
) -> dict[str, list[int]]:
    result = defaultdict(list)
    all_keys = set(d1) | set(d2)

    for key in all_keys:
        list1 = d1.get(key, [])
        list2 = d2.get(key, [])
        # merge keeps the result sorted if inputs are sorted
        result[key] = list(merge(list1, list2))

    return dict(result)


def truncate(sequence, length):
    seqs = [sequence[i : i + length] for i in range(0, len(sequence), length)]
    return seqs


def truncate_and_pad_to_long_tensor(
    list_of_sequences, sequence_length, padding_value=0
):
    trunced = [truncate(s, sequence_length) for s in list_of_sequences]
    list_of_pt = [torch.tensor(y) for r in trunced for y in r]
    return pad_sequence(list_of_pt, batch_first=True, padding_value=padding_value)
