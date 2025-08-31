from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
#import matplotlib.pyplot as plt

train = Path("/mnt/media/datasets/raw/yahoo_train.txt").read_text()
test = Path("/mnt/media/datasets/raw/yahoo_test.txt").read_text()

np.random.seed(42**2)

def invert(mp: dict):
    return {v:k for k, v in mp.items()}



def filter_dataset(ftr, lbl):
    keep_ftr = []
    keep_lbl = []
    for f, l in zip(ftr, lbl):
        if len(f) == 0:
            continue
        keep_ftr.append(f)
        keep_lbl.append(l)
    return keep_ftr, keep_lbl


def parse_dataset(src):
    labels = defaultdict(dict)
    features = set()
    for line in src.splitlines():
        index, target, value = map(int, line.split())
        labels[index][target] = value
        features.add(target)

    return labels, features


def make_labeled_data(raw: dict, fm, lm):
    inst_ftr = []
    inst_lbl = []
    for inst in raw.values():
        i_f = {}
        i_l = {}
        for key, val in inst.items():
            if key in fm:
                i_f[5*fm[key] + val] = 1.0
            elif key in lm:
                i_l[lm[key]] = val >= 4
                
        inst_ftr.append(i_f)
        inst_lbl.append(i_l)
    return inst_ftr, inst_lbl


def count_positives(lbl, kind=True):
    c = Counter()
    for l in lbl:
        c.update([k for k, v in l.items() if v is kind])
    hist = np.zeros(max(c)+1, dtype=int)
    for k, v in c.items():
        hist[k] = v
    return hist


FTRSEL = 500
train, features = parse_dataset(train)
test, features = parse_dataset(test)

features = np.asarray(list(features))
np.random.shuffle(features)
feature_map = dict((v, k) for (k, v) in enumerate(sorted(features[:FTRSEL])))
label_map = dict((v, k) for (k, v) in enumerate(sorted(features[FTRSEL:])))

train_ftr, train_lbl = make_labeled_data(train, feature_map, label_map)
test_ftr, test_lbl = make_labeled_data(test, feature_map, label_map)


# remove all labels for which we do not have a single test positive
test_lbl_dist = count_positives(test_lbl)
train_lbl_dist = count_positives(train_lbl)
valid = np.logical_and(test_lbl_dist > 0, train_lbl_dist > 0)
remove_labels = np.where(np.logical_not(valid))[0]
id_to_f = invert(label_map)
remove_labels = {id_to_f[r] for r in remove_labels}
label_map = dict((v, k) for (k, v) in enumerate(sorted(set(features[FTRSEL:]) - remove_labels)))

train_ftr, train_lbl = make_labeled_data(train, feature_map, label_map)
test_ftr, test_lbl = make_labeled_data(test, feature_map, label_map)
test_lbl_dist = count_positives(test_lbl)
train_lbl_dist = count_positives(train_lbl)

test_lbl_neg = count_positives(test_lbl, False)

true_label_prior = test_lbl_dist / (test_lbl_neg + test_lbl_dist)
observed_label_prior = train_lbl_dist / len(train_lbl)
propensity = observed_label_prior / true_label_prior
#print(propensity)


final_labels = np.where(propensity > 0.01)[0]
id_to_f = invert(label_map)
final_labels_orig = set(id_to_f[i] for i in final_labels)
final_features_orig = set(range(1000)) - final_labels_orig

feature_map = dict((v, k) for (k, v) in enumerate(sorted(final_features_orig)))
label_map = dict((v, k) for (k, v) in enumerate(sorted(final_labels_orig)))


# and one last time
train_ftr, train_lbl = make_labeled_data(train, feature_map, label_map)
test_ftr, test_lbl = make_labeled_data(test, feature_map, label_map)

test_ftr, test_lbl = filter_dataset(test_ftr, test_lbl)

test_lbl_dist = count_positives(test_lbl)
train_lbl_dist = count_positives(train_lbl)

test_lbl_neg = count_positives(test_lbl, False)

true_label_prior = test_lbl_dist / (test_lbl_neg + test_lbl_dist)
observed_label_prior = train_lbl_dist / len(train_lbl)
propensity = observed_label_prior / true_label_prior
#print(propensity)

#plt.hist(propensity)
#plt.show()

v_prop = test_lbl_dist / len(test_lbl) / (test_lbl_dist / (test_lbl_neg + test_lbl_dist))
#print(len(v_prop))


# ok, finally generate the train/test file
def write_dataset(ftr, lbl, fname):
    mf = 0
    ni = 0
    idf = Counter()
    for f, l in zip(ftr, lbl):
        if len(f) == 0:
            continue
        idf.update(f.keys())
        mf = max([mf] + list(f.keys()))
        ni += 1

    idf = {k: np.log(ni / v) for k, v in idf.items()}

    with open(fname, "w") as tgt:
        tgt.write(f"{ni} {mf + 1} {len(v_prop)}\n")
        for f, l in zip(ftr, lbl):
            if len(f) == 0:
                continue
            first = True
            for i, v in l.items():
                if v:
                    if not first:
                        tgt.write(",")
                    tgt.write(f"{i}")
                    first = False

            # TF-IDF features
            for k, v in f.items():
                tgt.write(f" {k}:{v*idf[k]}")
            tgt.write("\n")


# ok, finally generate the train/test file
def write_dataset_np(ftr, lbl, prop, fname):
    mf = 0
    ni = 0
    for f, l in zip(ftr, lbl):
        mf = max([mf] + list(f.keys()))
        ni += 1

    feature_mat = np.zeros((ni, mf+1))
    label_mat = np.zeros((ni, len(v_prop)))

    for i, (f, l) in enumerate(zip(ftr, lbl)):
        for k, v in l.items():
            label_mat[i, k] = 1 if v else -1

        for k, v in f.items():
            feature_mat[i, k] = 1.0

    np.savez(fname, features=feature_mat, labels=label_mat, propensity=prop)


print(len(test_lbl))
write_dataset(train_ftr, train_lbl, "yahoo_train.txt")
write_dataset(test_ftr, test_lbl, "yahoo_test.txt")
np.save("yahoo_train_prop.npy", np.minimum(propensity, 1.0))
np.save("yahoo_test_prop.npy", (test_lbl_neg + test_lbl_dist) / len(test_lbl))

write_dataset_np(train_ftr, train_lbl, np.minimum(propensity, 1.0), "yahoo_train.npz")
write_dataset_np(test_ftr, test_lbl, (test_lbl_neg + test_lbl_dist) / len(test_lbl), "yahoo_test.npz")

preds = np.load("predictions.npz")
pos = 0
neg = 0
unk = 0
for s, (gt, pred) in enumerate(zip(test_lbl, preds["arr_0"])):
    tk = np.argmax(pred)
    if s == 49:
        print(gt)
        print(tk)
    if tk in gt:
        if gt[tk]:
            print(s)
            pos += 1
        else:
            neg += 1
    else:
        unk += 1

print(pos)
print(pos / (pos + neg))
print(pos / (pos + neg + unk) / ((pos + neg) / unk))
print((pos + neg) / unk)
#print(gt, np.argmax(pred))
