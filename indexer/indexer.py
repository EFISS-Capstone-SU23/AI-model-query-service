import json
import logging
import os
import sys
import argparse
from utils.logger import setup_logging
from pprint import pp, pprint

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
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    ):
        """
        Args:
            model_path: path to the model
            database: list of image path
            new_index_database_version: version of the new index
            index_mode: mode of the index
        """
        logging.info(f"Begin indexing...")
        begin_time = time()
        # Getting Datetime from timestamp
        date_time = datetime.fromtimestamp(time()).strftime("%d/%m/%Y %H:%M:%S")
        logging.info(f"Datetime from timestamp: {date_time}")

        # create index
        hashcodes = self.compute_hashcodes(
            model_path=model_path, image_size=image_size, database=database,
        )
        # dump index
        self.dump_index(
            hashcodes=hashcodes,
            database=database,
            dump_index_path=self.configs['dump_index_path'],
            new_index_database_version=new_index_database_version,
            index_mode=index_mode,
            date_time=date_time,
            image_size=image_size,
            model_path=model_path,
        )

        logging.info(f"Finish indexing new_index_database_version: {new_index_database_version}")
        elapsed_time = time() - begin_time
        logging.info(f"Elapsed time: {elapsed_time}")
        return dict(
            result="success",
            index_database_version=new_index_database_version,
            timestamp=date_time,
            elapsed_time=elapsed_time,
        )

    def compute_hashcodes(
        self,
        model_path: str,
        database: List[str],
        image_size: int = 256,
    ):
        """
        Args:
            model_path: torchscript model path
            database: List of image path
        """
        logging.info("Begin computing hashcodes")
        logging.info(f"Loading Model from path: {model_path}")
        # load model
        model = torch.jit.load(model_path, map_location=self.configs["device"])
        model.eval()

        # create data
        dataset = DeepHashingDataset(database, transform=A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]))

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            shuffle=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        logging.info("Compute hashcodes...")
        # compute hashcodes
        hashcodes = []
        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc="Compute hashcodes",
                total=len(dataloader),
                ascii=True,
                ncols=100,
                disable=False
            )
            for batch in pbar:
                batch = batch.to(self.configs["device"])
                hashcode = model(batch)
                hashcodes.append(hashcode.cpu())
        hashcodes = torch.cat(hashcodes, dim=0)
        self.configs["hashcode_length"] = hashcodes.shape[1]
        logging.info(f"Hashcodes shape: {hashcodes.shape}")
        del model, dataset, dataloader
        hashcodes = self.convert_int(hashcodes)
        logging.info("Finish computing hashcodes")
        return hashcodes

    def dump_index(
        self,
        model_path: str,
        hashcodes: np.ndarray,
        database: List[str],
        dump_index_path: str,
        new_index_database_version: str,
        index_mode: str = "default",
        date_time: str = "01/01/2021 00:00:00",
        image_size: int = 256,
    ):
        """
        Args:
            hashcodes: hashcodes
            database: list of image path
            dump_index_path: path to dump the index
            new_index_database_version: version of the new index
            index_mode: mode of the index. If mode is "default", use faiss.IndexBinaryFlat, if mode is "ivf", use faiss.IndexBinaryIVF.
            date_time: datetime of the index
        """
        logging.info("Dump remap_index_to_img_path_dict...")
        remap_index_to_img_path_dict = self.create_bi_directional_dictionary(database)
        logging.info("Done!")
        logging.info("Dump index...")
        if index_mode == "default":
            logging.info("Use faiss.IndexBinaryFlat")
            # create index
            index = faiss.IndexBinaryFlat(self.configs["hashcode_length"])
            index.add(hashcodes)
        elif index_mode == "ivf":
            logging.info("Use faiss.IndexBinaryIVF")
            # create index
            quantizer = faiss.IndexBinaryFlat(self.configs["hashcode_length"])
            index = faiss.IndexBinaryIVF(quantizer, self.configs["hashcode_length"], 100)
            index.train(hashcodes)
            index.add(hashcodes)
        else:
            raise ValueError(f"index_mode {index_mode} is not supported")

        logging.info("Done adding index")
        # dump index
        index_path = os.path.join(dump_index_path, new_index_database_version)
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index_binary(index, os.path.join(index_path, "index.bin"))

        # update configs
        self.configs["index_database_version"] = new_index_database_version
        self.configs["index_mode"] = index_mode
        self.configs["index_path"] = index_path
        self.configs["index_datetime"] = date_time
        self.configs["image_size"] = image_size
        self.configs["model_path"] = model_path
        self.configs["model_name"] = model_path.split("/")[-1]
        with open(os.path.join(dump_index_path, new_index_database_version, "config.json"), "w") as f:
            json.dump(self.configs, f)
        with open(os.path.join(dump_index_path, new_index_database_version, "remap_index_to_img_path_dict.json"), "w") as f:
            json.dump(remap_index_to_img_path_dict, f)

        logging.info("Dump index successfully")
        logging.info(f"Configs: {self.configs}")
    
    @staticmethod
    def create_bi_directional_dictionary(database: List[str]):
        remap_index_to_img_path_dict = {}
        remap_img_path_to_index_dict = {}
        for index, img_path in enumerate(database):
            # replace 'resize_224x224' with output
            img_path = img_path.replace("resize_224x224", "product_images")
            img_path = img_path.replace("resize_300x300", "product_images")
            img_path = img_path.replace("resize_640x640", "product_images")
            img_path = img_path.replace("resize_600x600", "product_images")
            img_path = img_path.replace("data/shopee_crop_yolo/", "")
            
            remap_index_to_img_path_dict[str(index)] = img_path
            remap_img_path_to_index_dict[img_path] = str(index)
        
        return {
            "remap_index_to_img_path_dict": remap_index_to_img_path_dict,
            "remap_img_path_to_index_dict": remap_img_path_to_index_dict,
        }

    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        del codes
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out

def main(args):
    new_index_database_version = args.new_index_database_version
    logging.info(f"Begin indexing new_index_database_version: {new_index_database_version}")
    with open(args.database, "r") as f:
        database = f.read().splitlines()
    model_path = args.model_path
    index_mode = args.index_mode
    image_size = args.image_size

    if image_size is None:
        if "tf_efficientnetv2_b3" in model_path or "medium" in model_path:
            image_size = 300
        elif "tf_efficientnet_b7_ns" in model_path or "large" in model_path:
            image_size = 600
        elif "mobilenetv3small" in model_path or "small" in model_path:
            image_size = 224
        else:
            raise ValueError("image_size is not specified")
    logging.info(f"image_size: {image_size}")

    indexer_service = Indexer(dict(
        dump_index_path=args.dump_index_path,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    ))

    results = indexer_service.create_index(
        model_path=model_path,
        image_size=image_size,
        new_index_database_version=new_index_database_version,
        database=database,
        index_mode=index_mode,
    )

    logging.info(f"Return results: {results}")

def run(_args=None):
    if _args is None:
        __args = sys.argv[1:]
    else:
        __args = _args

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', help="config file", type=str, default='config.json')
    parser.add_argument('--dump_index_path', help="path to dump index", type=str, default='index')
    parser.add_argument('--device', help="device", type=str, default='cpu')
    parser.add_argument('--num_workers', help="num_workers", type=int, default=4)
    parser.add_argument('--batch_size', help="batch_size", type=int, default=64)
    parser.add_argument('--model_path', help="model_path", type=str, default='torchscripts_models/relahash_tf_efficientnetv2_b3.pt')
    parser.add_argument('--image_size', help="image_size", type=int, default=None, required=False)
    parser.add_argument('--new_index_database_version', help="new_index_database_version", type=str, default='1.2.0')
    parser.add_argument('--database', help="database", type=str, required=True)
    parser.add_argument('--index_mode', help="index_mode", type=str, default='default')
    args = parser.parse_args(__args)
    setup_logging('indexer.log')
    pprint(args)
    main(args)

if __name__ == '__main__':
    command = """
    python indexer/main.py
    """
    run()