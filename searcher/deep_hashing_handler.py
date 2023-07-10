import torch
import base64
import numpy as np
import faiss
from PIL import Image
from ts.torch_handler.vision_handler import VisionHandler
import io
import zipfile
import os
import json
import logging
from typing import Dict, List, Tuple
from torchvision import transforms
import torch.cuda.amp as amp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


logger = logging.getLogger(__name__)

class DeepHashingHandler(VisionHandler):

    @staticmethod
    def data_uri_to_cv2_img(uri: str) -> np.ndarray:
        # https://stackoverflow.com/a/42538142/11806050
        encoded_data = uri.split(',')[-1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def __init__(self):
        super(DeepHashingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logging.info(f"Using device: {self.device}")

        if os.path.isfile(os.path.join(model_dir, "module.zip")):
            with zipfile.ZipFile(model_dir + "/module.zip", "r") as zip_ref:
                zip_ref.extractall(model_dir)
        
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            raise Exception("Missing the config.json file.")

        logger.info(f"Loading model from {model_pt_path}")
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()
        logger.info(f'Model loaded successfully from {model_pt_path}: {self.model}')

        # Load the index
        logger.info(f"Loading index ...")
        self.index = faiss.read_index_binary(os.path.join(model_dir, "index.bin"))

        if os.path.isfile(os.path.join(model_dir, "remap_index_to_img_path_dict.json")):
            with open(os.path.join(model_dir, "remap_index_to_img_path_dict.json")) as f:
                self.remap_index_to_img_path_dict = json.load(f)["remap_index_to_img_path_dict"]
        else:
            raise Exception("Missing the remap_index_to_img_path_dict.json file.")

        image_size = self.setup_config["image_size"]
        self.image_processing = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def transform(self, image):
        image = self.image_processing(image=image)['image'].float()
        return image

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor
                - row: {
                    "topk": 10,
                    "image": "base64 encoded image"
                }
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []
        topk_batch: List[int] = []
        debug = False

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            req = row.get("data") or row.get("body")
            logger.info(f"req: {str(req)[:200]}")
            if req is None:
                logger.error("Malformed input data!")
            if isinstance(req, str):
                req = json.loads(req)
            elif isinstance(req, (bytearray, bytes)):   
                req = json.loads(req.decode("utf-8"))
            elif isinstance(req, dict):
                pass
            else:
                logger.error(f"Unknown input type: {type(req)}")
            topk: int = req.get("topk", 10)
            image: str = req.get("image")
            debug: bool = req.get("debug", False)  # NOTE: debug mode is set for all images in a batch

            image = self.data_uri_to_cv2_img(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)

            images.append(image)
            topk_batch.append(topk)

        return torch.stack(images).to(self.device), topk_batch, debug

    def inference(self, batch):
        """
        The Inference Function receives the pre-processed input in form of Tensor
        Args:
            batch (torch.tensor): list of images, topk_batch, debug
                - images: torch.tensor of shape (batch_size, 3, image_size, image_size)
                - topk_batch (List[int]): list of topk for each image in the batch
                - debug: bool, whether to return debug information
        Returns:
            D (torch.tensor): distance matrix of shape (batch_size, topk)
            I (torch.tensor): index matrix of shape (batch_size, topk)
            debug (bool): whether to return debug information
        """
        img_tensor, topk_batch, debug = batch
        logger.info(f"img_tensor.shape: {img_tensor.shape}")
        logger.info(f"topk_batch: {topk_batch}")
        logger.info(f"img_tensor.device: {img_tensor.device}")
        with torch.no_grad(), amp.autocast():
            img_tensor = img_tensor.to(self.device)
            features = self.model(img_tensor)
        logging.info(f"Hashcodes shape: {features.shape}")
        hashcodes = self.convert_int(features)
        logging.info("Finish computing hashcodes")

        if len(set(topk_batch)) == 1:
            # all topk are the same, we can use batch search
            D, I = self.index.search(hashcodes, topk_batch[0])
        else:
            I = []
            for i, topk in enumerate(topk_batch):
                D, _I = self.index.search(hashcodes[i, :].reshape(1, -1), topk)
                logger.info(f"Top {topk} similar images for image {i}: {_I}")
                logger.info(f"Top {topk} distances for image {i}: {D}")
                logger.info(f"I.shape: {_I.shape}")
                I.append(_I[0])
        
        # TODO:
        # If there is any images that have the same distance as the top 1 result,
        # we will use features to compute the distance and sort them.
        # This is ensure that if the query exists in the database, it will be the top 1 result.

        return D, I, debug
    
    def postprocess(self, inference_output):
        """
        The post-process function receives the return value of the inference function.
        It performs post-processing on the raw output to convert it into a format that is
        easy for the user to understand.
        
        Args:
            D (torch.tensor): distance matrix of shape (batch_size, topk)
            I (torch.tensor): index matrix of shape (batch_size, topk)
            debug (bool): whether to return debug information
        
        Returns:
            responses (List[Dict]): list of responses
                - index_database_version (str): version of the index database
                - relevant (List[str]): list of relevant images, each image is a string of image path, sorted by relevance
                - distances (List[int]): list of distances. Only returned if debug is True
        """
        D, I, debug = inference_output
        logger.info(f"Postprocess: I: {I}")
        logger.info(f"Postprocess: D: {D}")
        img_paths: List[List[int]] = [[self.remap_index_to_img_path_dict[str(idx)] for idx in idxs] for idxs in I]
        if not debug:
            responses = [{
                "index_database_version": self.setup_config["index_database_version"],
                "relevant": img_path
            } for img_path in img_paths]
        else:
            responses = [{
                "index_database_version": self.setup_config["index_database_version"],
                "relevant": img_path,
                "distances": dists.tolist()
            } for img_path, dists in zip(img_paths, D)]
        logger.info(f"Postprocess: responses: {responses}")
        return responses

    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out
