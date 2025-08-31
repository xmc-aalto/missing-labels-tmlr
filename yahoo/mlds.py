import multiprocessing

import numpy as np
import tqdm
import dataclasses
from typing import Tuple, List, Union
from pathlib import Path


@dataclasses.dataclass
class DatasetBase:
    source: str
    num_documents: int
    num_features: int
    num_labels: int


@dataclasses.dataclass
class TextLineDataset(DatasetBase):
    labels: List[np.ndarray]
    features: List[np.ndarray]


@dataclasses.dataclass
class SparseDataset(DatasetBase):
    indices: np.ndarray
    values: np.ndarray
    positives: np.ndarray


def _parse_labels(lbl: str):
    if len(lbl) == 0:
        return np.array([], dtype=np.int32)
    else:
        return np.fromiter(map(int, lbl.split(",")), dtype=np.int32)


def _parse_features(ftr: str):
    num_features = ftr.count(':')
    result = np.empty(shape=(num_features,), dtype=np.dtype('i4, f4'))
    for k, feature in enumerate(ftr.split(" ")):
        idx, _, val = feature.partition(":")
        value = float(val)
        index = int(idx)
        result[k] = (index, value)
    return result


def read_data(file_name: str) -> TextLineDataset:
    in_file = open(file_name, "r")

    header = in_file.readline().rstrip('\n').split(" ")
    num_docs = int(header[0])
    num_features = int(header[1])
    num_labels = int(header[2])
    ds = TextLineDataset(Path(file_name).name, num_docs, num_features, num_labels, [], [])

    labels = []
    features = []

    for _ in tqdm.tqdm(range(num_docs), desc=f'Reading {file_name}'):
        sample = in_file.readline()[:-1].split(" ", 1)
        labels.append(sample[0])
        features.append(sample[1])

    pool = multiprocessing.Pool(processes=8)
    labels = pool.imap(_parse_labels, labels, chunksize=4096)
    features = pool.imap(_parse_features, features, chunksize=4096)
    ds.labels = list(labels)
    ds.features = list(features)
    return ds


def write_data(file_name: str, ds: TextLineDataset):
    out_file = open(file_name, "w")
    out_file.write(f"{ds.num_documents} {ds.num_features} {ds.num_labels}\n")
    for labels, features in tqdm.tqdm(zip(ds.labels, ds.features), desc=f'Writing dataset {file_name}'):
        out_file.write(",".join(map(str, labels)))
        for f in features:
            out_file.write(f"{f[1]}:{f[2]}")
        out_file.write("\n")


def to_sparse(ds: TextLineDataset) -> SparseDataset:
    num_features = sum(map(len, ds.features))
    num_positives = sum(map(len, ds.labels))
    all_indices = np.empty(shape=(num_features, 2), dtype=np.int64)
    all_positives = np.empty(shape=(num_positives, 2), dtype=np.int64)
    all_values = np.empty(shape=(num_features,), dtype=np.float32)
    ftr_idx = 0
    lbl_idx = 0
    for doc, (lbl, ftr) in enumerate(tqdm.tqdm(zip(ds.labels, ds.features))):
        all_indices[ftr_idx:ftr_idx+len(ftr), 0] = doc
        all_indices[ftr_idx:ftr_idx+len(ftr), 1] = ftr['f0']
        all_values[ftr_idx:ftr_idx+len(ftr)] = ftr['f1']
        ftr_idx += len(ftr)
        all_positives[lbl_idx:lbl_idx+len(lbl), 0] = doc
        all_positives[lbl_idx:lbl_idx+len(lbl), 1] = lbl
        lbl_idx += len(lbl)

    return SparseDataset(source=ds.source, num_documents=ds.num_documents, num_labels=ds.num_labels,
                         num_features=ds.num_features, indices=all_indices, values=all_values, positives=all_positives)


def to_tensors(ds: SparseDataset):
    import tensorflow as tf
    features = tf.SparseTensor(ds.indices, ds.values, (ds.num_documents, ds.num_features))
    labels = tf.SparseTensor(ds.positives, tf.ones(len(ds.positives), dtype=tf.float32),
                             (ds.num_documents, ds.num_labels))
    return tf.sparse.reorder(features), tf.sparse.reorder(labels)


def count_labels(ds: TextLineDataset) -> np.array:
    """
    Counts how often each label occurs
    """
    label_frequency = np.zeros(ds.num_labels, dtype=np.int32)
    for labels in tqdm.tqdm(ds.labels, desc=f'Counting labels for {ds.source}'):
        for label in labels:
            label_frequency[label] += 1
    return label_frequency


def remap_labels(ds: TextLineDataset, remapping: dict):
    """
    Remaps all label ids based on the `remapping` dict. Any label that does not occur in that dict is dropped.
    """
    for i, labels in tqdm.tqdm(enumerate(ds.labels), desc=f'Remapping labels for {ds.source}',
                               total=ds.num_documents):
        new_labels = []
        for label in labels:
            if label in remapping:
                new_labels.append(remapping[label])
        ds.labels[i] = np.array(new_labels)
        ds.num_labels = len(remapping)


def split_dataset(ds: TextLineDataset, ratio: float):
    all_indices = np.arange(ds.num_documents)
    np.random.shuffle(all_indices)
    # make a lookup array which tells us
    doc_lookup = np.ones((ds.num_documents,), dtype=np.bool)
    for i in range(int(ds.num_documents * ratio)):
        doc_lookup[all_indices[i]] = False

    ds1 = TextLineDataset(source=ds.source+"@1", num_documents=np.sum(doc_lookup == 0),
                          num_features=ds.num_features, num_labels=ds.num_labels,
                          labels=[], features=[])

    ds2 = TextLineDataset(source=ds.source+"@2", num_documents=np.sum(doc_lookup == 1),
                          num_features=ds.num_features, num_labels=ds.num_labels,
                          labels=[], features=[])

    for i in range(ds.num_documents):
        if doc_lookup[i] == 0:
            ds1.labels.append(ds.labels[i])
            ds1.features.append(ds.features[i])
        else:
            ds2.labels.append(ds.labels[i])
            ds2.features.append(ds.features[i])

    return ds1, ds2
