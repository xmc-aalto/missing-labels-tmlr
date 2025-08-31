# Unbiased Loss Functions for Multilabel Classification with Missing Labels
https://openreview.net/forum?id=hMq1hUhLqp

+ [eval](eval) contains the code for generating results evaluting recall-at-k on fully artificial data.
+ [train](train) contains the code for generating the semi-artificial dataset, where [train/keep_top.py](keep_top.py) is used to remove all but the top-k labels from an XMC dataset, and [train/runner.py](runner.py) runs a sweep over different regularizations for training with unbiased/upper-bound losses. The calculations of the losses themselves are defined in [train/losses.py](losses.py). Note that missing labels are induced inside the runner script, instead of during data-preprocessing, because we want to calculate both original loss function on original data, and ubiased/upper-bound loss on missing-label data.
+ [yahoo](yahoo) contains the code for the yahoo-music-r3 experiment. It follows the same structure as train, but uses [yahoo/prep.py](prep.py) to generate the training data from the original [music rating data](https://web.archive.org/web/20250403012444/https://webscope.sandbox.yahoo.com/).


