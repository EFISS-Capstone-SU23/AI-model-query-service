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
from time import time
from datetime import datetime


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
        begin_time = time()
        # Getting Datetime from timestamp
        date_time = datetime.fromtimestamp(time()).strftime("%d/%m/%Y %H:%M:%S")
        logging.info("Datetime from timestamp:", date_time)

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
            date_time=date_time,
        )

        return dict(
            result="success",
            index_database_version=new_index_database_version,
            timestamp=date_time,
            elapsed_time=time() - begin_time,
        )

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

    def dump_index(
        self,
        hashcodes: np.ndarray,
        dump_index_path: str,
        new_index_database_version: str,
        index_mode: str = "default",
        date_time: str = "01/01/2021 00:00:00",
    ):
        """
        Args:
            hashcodes: hashcodes
            dump_index_path: path to dump the index
            new_index_database_version: version of the new index
            index_mode: mode of the index. If mode is "default", use faiss.IndexBinaryFlat, if mode is "ivf", use faiss.IndexBinaryIVF.
            date_time: datetime of the index
        """
        logging.info("Dump index...")
        if index_mode == "default":
            logging.info("Use faiss.IndexBinaryFlat")
            # create index
            index = faiss.IndexBinaryFlat(hashcodes.shape[-1])
            index.add(hashcodes)
        elif index_mode == "ivf":
            logging.info("Use faiss.IndexBinaryIVF")
            # create index
            quantizer = faiss.IndexBinaryFlat(hashcodes.shape[-1])
            index = faiss.IndexBinaryIVF(quantizer, hashcodes.shape[-1], 100)
            index.train(hashcodes)
            index.add(hashcodes)
        else:
            raise ValueError(f"index_mode {index_mode} is not supported")

        # dump index
        index_path = os.path.join(dump_index_path, new_index_database_version)
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index_binary(index, os.path.join(index_path, "index.bin"))

        # update configs
        self.configs["index_database_version"] = new_index_database_version
        self.configs["index_mode"] = index_mode
        self.configs["index_path"] = index_path
        self.configs["index_datetime"] = date_time
        with open(os.path.join(dump_index_path, new_index_database_version, "configs.json"), "w") as f:
            json.dump(self.configs, f)

        logging.info("Dump index successfully")
        logging.info(f"Configs: {self.configs}")

    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out
