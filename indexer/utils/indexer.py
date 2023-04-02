import json
import logging
import os

from PIL import Image
import faiss
import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm
from torchvision import transforms
from typing import List, Dict, Tuple, Union, Optional, Any, Literal
from utils.datasets import DeepHashingDataset


class Indexer:
    def __init__(self, configs):
        self.configs = configs

    def create_index(
        self,
        model_path: str,
        database: List[str],
        image_size: int = 256,
        new_index_database_version: str = "1.0.0",
        index_mode: str = "default",
        dump_index_path: str = "index",
    ):
        """
        Args:
            model_path: path to the model
            database: list of image path
            new_index_database_version: version of the new index
            index_mode: mode of the index
            dump_index_path: path to dump the index
        """
        # create index
        hashcodes = self.compute_hashcodes(
            model_path=model_path, image_size=image_size, database=database,
        )
        # dump index
        self.dump_index(
            hashcodes=hashcodes,
            dump_index_path=dump_index_path,
            new_index_database_version=new_index_database_version,
            index_mode=index_mode,
        )
        # return results
        results = dict(
            result="success",
            previous_index_database_version="1.1.0",
            index_database_version="1.2.0",
            timestamp="2020-05-02 12:00:00",
            elapsed_time=100,  # seconds
        )
        return results

    def compute_hashcodes(
        self,
        model_path: str,
        database: List[str],
        index_mode: str = "default",
        image_size: int = 256,
    ):
        """
        Args:
            model_path: torchscript model path
            database: List of image path
        """
        # load model
        model = torch.jit.load(model_path, map_location=self.configs["device"])
        model.eval()

        # create data
        dataset = DeepHashingDataset(
            database, transform=transforms.Compose([transforms.Resize(image_size),])
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            shuffle=False,
            pin_memory=True,
        )

        # compute hashcodes
        hashcodes = []
        with torch.no_grad(), amp.autocast():
            pbar = tqdm(
                dataloader,
                desc="Compute hashcodes",
                total=len(dataloader),
                ascii=True,
                ncols=100,
            )
            for batch in pbar:
                batch = batch.to(self.configs["device"])
                hashcode = model(batch)
                hashcodes.append(hashcode.cpu())
        hashcodes = torch.cat(hashcodes, dim=0)
        hashcodes = self.convert_int(hashcodes)
        return hashcodes

    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out
