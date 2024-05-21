import sys
import random
import logging

import numpy as np
import torch
import pickle
from tqdm import tqdm

def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    print("Seed set", seed)


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def load_mosei(emo="7class"):
    unsplit = load_pkl("data/mosei/mosei_data.pkl")

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    valVid = list(unsplit["valVid"])
    testVid = list(unsplit["testVid"])
    
    spliter = {
        "train": trainVid,
        "dev": valVid,
        "test": testVid
    }

    for split in data:
        for uid in tqdm(spliter[split], desc=split):
            data[split].append(
                {
                    "uid" : uid,
                    "speakers" : [0] * len(unsplit["speaker"][uid]),
                    "labels" : unsplit['label'][emo][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
    
    return data