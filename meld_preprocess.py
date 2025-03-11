import argparse
from numpy.lib.twodim_base import diag

from tqdm import tqdm
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import MITPA

parser = argparse.ArgumentParser(description="preprocess.py")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["iemocap", "iemocap_4", "meld"],
    default= "meld",
    help="Dataset name.",
)

parser.add_argument(
    "--data_dir", type=str, default="./data", help="Dataset directory"
)
parser.add_argument(
    "--data_root",
    type=str,
    default="/home/tuanma/tuanma/LPE_ERC/data/meld",
)
parser.add_argument("--seed", type=int, default=24, help="Random seed.")
args = parser.parse_args()

log = MITPA.utils.get_logger()
sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")
seed = 42


def get_meld():
    MITPA.utils.set_seed(args.seed)

    if args.dataset == "meld":
        (
            utt_ids,
            speakers,
            emo_labels,
            sent_labels,
            embedding_1,  # data[4]
            embedding_2,  # data[5]
            embedding_3,  # data[6]
            embedding_4,  # data[7]
            embedding_5,  # data[8]
            embedding_6,  # data[9]
            raw_text,
            train_ids,
            test_ids,
            NaN,
        ) = pickle.load(
            open(args.data_root + "/meld_multi_features_cfn-esa.pkl", "rb"), encoding="latin1"
        )

    train, dev, test = [], [], []
    dev_size = int(len(train_ids) * 0.1)
    train_vids, dev_vids = list(train_ids)[dev_size:], list(train_ids)[:dev_size]
    test_vids = test_ids
    # Within the function get_meld():

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            {
                "vid": vid,
                "speakers": speakers[vid],
                "labels": emo_labels[vid],
                "audio": [np.array(a, dtype=np.float32) for a in embedding_5[vid]],  # Explicitly set dtype
                "visual": [np.array(v, dtype=np.float32) for v in embedding_6[vid]],  # Explicitly set dtype
                "text": [np.array(a, dtype=np.float32) for a in embedding_1[vid]],
                "sentence": raw_text[vid],
            }
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            {
                "vid": vid,
                "speakers": speakers[vid],
                "labels": emo_labels[vid],
                "audio": [np.array(a, dtype=np.float32) for a in embedding_5[vid]],  # Explicitly set dtype
                "visual": [np.array(v, dtype=np.float32) for v in embedding_6[vid]],  # Explicitly set dtype
                "text": [np.array(a, dtype=np.float32) for a in embedding_1[vid]],
                "sentence": raw_text[vid],
            }
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            {
                "vid": vid,
                "speakers": speakers[vid],
                "labels": emo_labels[vid],
                "audio": [np.array(a, dtype=np.float32) for a in embedding_5[vid]],  # Explicitly set dtype
                "visual": [np.array(v, dtype=np.float32) for v in embedding_6[vid]],  # Explicitly set dtype
                "text": [np.array(a, dtype=np.float32) for a in embedding_1[vid]],
                "sentence": raw_text[vid],
            }
        )

        # log.info("train vids:")
        # log.info(sorted(train_vids))
        # log.info("dev vids:")
        # log.info(sorted(dev_vids))
        # log.info("test vids:")
        # log.info(sorted(test_vids))

    return train, dev, test


train, dev, test = get_meld()

data = {"train": train, "dev": dev, "test": test}
MITPA.utils.save_pkl(data, "/home/tuanma/tuanma/LPE_ERC/data/meld/data_meld.pkl")

# log.info("number of train samples: {}".format(len(train)))
# log.info("number of dev samples: {}".format(len(dev)))
# log.info("number of test samples: {}".format(len(test)))
