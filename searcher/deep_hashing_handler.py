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


logger = logging.getLogger(__name__)

class DeepHashingHandler(VisionHandler):
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
        logger.info(f"Loading index from {self.setup_config['abs_index_path']}")
        self.index = faiss.read_index_binary(self.setup_config["abs_index_path"])

        if os.path.isfile(os.path.join(model_dir, "remap_index_to_img_path_dict.json")):
            with open(os.path.join(model_dir, "remap_index_to_img_path_dict.json")) as f:
                self.remap_index_to_img_path_dict = json.load(f)["remap_index_to_img_path_dict"]
        else:
            raise Exception("Missing the remap_index_to_img_path_dict.json file.")

        image_size = self.setup_config["image_size"]
        self.image_processing = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

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
        topk_batch = []
        debug = False

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            req = row.get("data") or row.get("body")
            req = json.loads(req)
            topk = req.get("topk", 10)
            image = req.get("image")
            debug = req.get("debug", False)
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)
            topk_batch.append(topk)

        return torch.stack(images).to(self.device), topk_batch, debug

    def inference(self, batch):
        img_tensor, topk_batch, debug = batch
        logger.info(f"img_tensor.shape: {img_tensor.shape}")
        logger.info(f"topk_batch: {topk_batch}")
        logger.info(f"img_tensor.device: {img_tensor.device}")
        with torch.no_grad(), amp.autocast():
            img_tensor = img_tensor.to(self.device)
            hashcodes = self.model(img_tensor)
        logging.info(f"Hashcodes shape: {hashcodes.shape}")
        features = self.convert_int(hashcodes)
        logging.info("Finish computing hashcodes")

        if len(set(topk_batch)) == 1:
            # all topk are the same, we can use batch search
            D, I = self.index.search(features, topk_batch[0])
        else:
            I = []
            for i, topk in enumerate(topk_batch):
                D, _I = self.index.search(features[i, :].reshape(1, -1), topk)
                logger.info(f"Top {topk} similar images for image {i}: {_I}")
                logger.info(f"Top {topk} distances for image {i}: {D}")
                logger.info(f"I.shape: {_I.shape}")
                I.append(_I[0])
        return D, I, debug
    
    def postprocess(self, inference_output):
        D, I, debug = inference_output
        logger.info(f"Postprocess: I: {I}")
        logger.info(f"Postprocess: D: {D}")
        img_paths = [[self.remap_index_to_img_path_dict[str(idx)] for idx in idxs] for idxs in I]
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
