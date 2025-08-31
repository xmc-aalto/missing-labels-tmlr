import numpy as np
from pathlib import Path
import argparse
import mlds
import pickle


def main():
    parser = argparse.ArgumentParser("forget")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--keep", type=int, required=True)

    args = parser.parse_args()
    keep_top_labels(args.train, args.test, destination=args.dest, keep=args.keep)


def keep_top_labels(train_file: str, test_file: str, destination: str, keep: int):

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    train_ds = mlds.read_data(train_file)
    test_ds = mlds.read_data(test_file)

    counts = mlds.count_labels(train_ds)

    # remove all the tail labels
    ordered = list(sorted(enumerate(counts), reverse=True, key=lambda x: x[1]))

    labels_to_retain = [label for label, count in ordered[:keep]]
    remapping = {old: new for (new, old) in enumerate(labels_to_retain)}

    mlds.remap_labels(train_ds, remapping)
    mlds.remap_labels(test_ds, remapping)

    pickle.dump(train_ds, open(destination / "original-train.pkl", "wb"))
    pickle.dump(test_ds, open(destination / "original-test.pkl", "wb"))


if __name__ == "__main__":
    main()
