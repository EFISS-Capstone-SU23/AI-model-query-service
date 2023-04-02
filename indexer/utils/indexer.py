import json
import logging
import os

from PIL import Image
import faiss
import numpy as np
import torch
from tqdm import tqdm

class Indexer:
    @staticmethod
    def create_index(
            model,
            database,
            new_index_database_version="1.0.0",
            index_mode="default",
            dump_index_path="index",
        ):
        ...
    
    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out
