import json
import logging
import os
import sys
import argparse
from utils.logger import setup_logging
from pprint import pp, pprint

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm
from time import time
from datetime import datetime
from ultralytics import YOLO
import cv2



class Cropper:
    def __init__(self, configs):
        self.configs = configs

        logging.info(f"Loading model from {configs['model_path']}")
        self.yolo_model = YOLO(configs["model_path"])
        self.yolo_model.to(configs["device"])
        logging.info(f"Model loaded")

    def crop_image(self, img: np.ndarray) -> list[np.ndarray]:
        """
        Crop the image using YOLOv8
        
        Args:
            img (np.ndarray): the image to crop
        
        Returns:
            np.ndarray: the cropped images, if any
        """
        result = self.yolo_model.predict(
            source=img,
            conf=0.05,
            device='0' if self.device.type == 'cuda' else 'cpu',
            save=False,
            verbose=False
        )[0]
        if len(result.boxes.xyxy) == 0:
            logging.info("No object detected")
            return []
        
        out: list[np.ndarray] = []
        for box in result.boxes.xyxy:
            x, y, _x, _y = list(box.int())
            out.append(result.orig_img[y:_y, x:_x])
        
        return out

    def crop(
        self,
        database: list[str],
        previous_index_database_version: str = "1.0.0",
    ):
        """
        Args:
            model_path: path to the model
            database: list of image path
            previous_index_database_version: version of the new index
        """
        logging.info(f"Begin indexing...")
        begin_time = time()
        # Getting Datetime from timestamp
        date_time = datetime.fromtimestamp(time()).strftime("%d/%m/%Y %H:%M:%S")
        logging.info(f"Datetime from timestamp: {date_time}")

        # Load previous database
        previous_database_path = os.path.join(self.configs["dump_index_path"], previous_index_database_version, "remap_index_to_img_path_dict.json")
        with open(previous_database_path, "r") as f:
            previous_database: list[str] = list(json.load(f)["remap_img_path_to_index_dict"].keys())
        
        logging.info(f"Found {len(previous_database)} images in previous database")
        logging.info(f"Found {len(database)} images in current database")
        # get different between previous database and current database
        different_database: list[str] = list(set(database) - set(previous_database))
        logging.info(f"-> Found {len(different_database)} different images")

        cropped_image_paths: list[str] = self._crop_to_sub_images(different_database)
        logging.info(f"Found {len(cropped_image_paths)} sub images")
        
        with open(previous_database_path, "r") as f:
            

        logging.info(f"Finish indexing previous_index_database_version: {previous_index_database_version}")
        elapsed_time = time() - begin_time
        logging.info(f"Elapsed time: {elapsed_time}")
        return dict(
            result="success",
            index_database_version=previous_index_database_version,
            timestamp=date_time,
            elapsed_time=elapsed_time,
        )
    
    def _crop_to_sub_images(self, database: list[str]) -> list[str]:
        """
        Crop the images in the database to sub images and save them to disk
        
        Args:
            database (list[str]): list of image path
        
        Returns:
            list[str]: list of sub image path
        """
        cropped_image_paths: list[str] = []
        logging.info(f"Begin cropping {len(database)} images")
        for img_path in tqdm(database):
            img = cv2.imread(img_path)
            img_name: str = img_path.split("/")[-1].split(".")[0]

            try:
                sub_images: list[np.ndarray] = self.crop_image(img)
            except Exception as e:
                logging.error(f"Error cropping image {img_path}: {e}")
                continue

            if len(sub_images) == 0:
                continue
            for i, sub_img in enumerate(sub_images):
                sub_img_path: str = f"{self.output_datadir}/{img_name}_crop{i}.jpg"
                cv2.imwrite(sub_img_path, sub_img)
                cropped_image_paths.append(sub_img_path)

        logging.info(f"Finish cropping {len(database)} images")

        return cropped_image_paths

def main(args):
    previous_index_database_version = args.previous_index_database_version
    logging.info(f"Begin cropping different between version {previous_index_database_version} and current version")
    with open(args.database, "r") as f:
        database = f.read().splitlines()
    model_path = args.model_path

    cropper_service = Cropper(dict(
        model_path=model_path,
        dump_index_path=args.dump_index_path,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    ))

    results = cropper_service.crop(
        previous_index_database_version=previous_index_database_version,
        database=database,
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
    parser.add_argument('--model_path', help="model_path", type=str, default='torchscripts_models/yolo/yolo.pt')
    parser.add_argument('--previous_index_database_version', help="previous_index_database_version", type=str, default='1.2.0')
    parser.add_argument('--database', help="database", type=str, required=True)
    args = parser.parse_args(__args)
    setup_logging('cropper.log')
    pprint(args)
    main(args)

if __name__ == '__main__':
    command = """
    python cropper/cropper.py
    """
    run()