import torch
import base64
import numpy as np
import faiss
from ts.torch_handler.vision_handler import VisionHandler
import zipfile
import os
import json
import logging
import cv2
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn as nn

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
        # serialized_file = self.manifest["model"]["serializedFile"]
        # model_pt_path = os.path.join(model_dir, serialized_file)
        model_pt_path = None

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

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model.classifier = nn.Identity()
        model.eval()
        model.to(self.device)

        self.model = model
        self.processor = processor

        logger.info(f'Model loaded successfully from {model_pt_path}: {self.model}')

        # Load the index
        logger.info(f"Loading index ...")
        self.index = faiss.read_index(os.path.join(model_dir, "index.bin"))

        if os.path.isfile(os.path.join(model_dir, "remap_index_to_img_path_dict.json")):
            with open(os.path.join(model_dir, "remap_index_to_img_path_dict.json")) as f:
                self.remap_index_to_img_path_dict = json.load(f)["remap_index_to_img_path_dict"]
        else:
           raise Exception("Missing the remap_index_to_img_path_dict.json file.")

        self.yolo_model = YOLO(os.path.join(model_dir, "yolo.pt"))
        self.yolo_model.to(self.device)

    def transform(self, image: np.ndarray) -> torch.Tensor:
        return self.image_processing(image=image)['image'].float()
    
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
            conf=0.2,
            device='0' if self.device.type == 'cuda' else 'cpu',
            save=False,
            verbose=False
        )[0]
        if len(result.boxes.xyxy) == 0:
            logger.info("No object detected")
            return []
        
        out: list[np.ndarray] = []
        for box in result.boxes.xyxy:
            x, y, _x, _y = list(box.int())
            out.append(result.orig_img[y:_y, x:_x])

            # NOTE: for now, we only return the first cropped image
            return out
        
        return out
    
    def tokenize(self, img: np.ndarray) -> dict[str, torch.Tensor]:
        """
        Tokenize the image using ViT
    
        Args:
            img (np.ndarray): the image to tokenize, which has been ran through cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        Returns:
            dict[str, torch.Tensor]: the tokenized image
        """
        inputs = self.processor(images=img, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze()
        return inputs

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (list): Input data from the request is in the form of a Tensor
                - row: {
                    "topk": 10,
                    "image": "base64 encoded image"
                }
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images: list[np.ndarray] = []
        topk_batch: list[int] = []

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

            image: np.ndarray = self.data_uri_to_cv2_img(image)
            _images: list[np.ndarray] = self.crop_image(image)

            if len(_images) == 0:
                logger.info("Falling back to the original image")
                _images.append(image)

            # NOTE: for now, we only use the first cropped image
            cropped_image = _images[0]

            cropped_image: np.ndarray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            # cropped_image: torch.Tensor = self.transform(cropped_image)

            images.append(cropped_image)
            topk_batch.append(topk)

        assert len(images) == 1, "Currently, we only support one image at a time, edit config.properties to change this behavior"
        
        return images, topk_batch

    def inference(self, batch):
        """
        The Inference Function receives the pre-processed input in form of Tensor
        Args:
            batch (torch.tensor): list of images, topk_batch
                - images: torch.tensor of shape (batch_size, 3, image_size, image_size)
                - topk_batch (list[int]): list of topk for each image in the batch
        Returns:
            D (torch.tensor): distance matrix of shape (batch_size, topk)
            I (torch.tensor): index matrix of shape (batch_size, topk)
        """
        img_tensor, topk_batch = batch
        logger.info(f"img_tensor.shape: {img_tensor.shape}")
        logger.info(f"topk_batch: {topk_batch}")
        logger.info(f"img_tensor.device: {img_tensor.device}")
        img_tensor: np.ndarray = img_tensor[0]
        topk: int = topk_batch[0]

        inputs = self.processor(images=img_tensor, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            features = self.model(**inputs).logits
        logging.info(f"Features shape: {features.shape}")  # (1, 768)
        logging.info("Finish computing features")

        if len(set(topk_batch)) == 1:
            # all topk are the same, we can use batch search
            D, I = self.index.search(features, topk_batch[0] * 2)
            # TODO: topk * 4 to ensure there are too few images after filtered by product
        else:
            raise NotImplementedError("Currently, we only support the case where all topk are the same")
            I: list = []
            for i, topk in enumerate(topk_batch):
                D, _I = self.index.search(features[i, :].reshape(1, -1), topk)
                logger.info(f"Top {topk} similar images for image {i}: {_I}")
                logger.info(f"Top {topk} distances for image {i}: {D}")
                logger.info(f"I.shape: {_I.shape}")
                I.append(_I[0])
        
        # TODO:
        # If there is any images that have the same distance as the top 1 result,
        # we will use features to compute the distance and sort them.
        # This is ensure that if the query exists in the database, it will be the top 1 result.

        return D, I
    
    def postprocess(self, inference_output):
        """
        The post-process function receives the return value of the inference function.
        It performs post-processing on the raw output to convert it into a format that is
        easy for the user to understand.
        
        Args:
            D (torch.tensor): distance matrix of shape (batch_size, topk)
            I (torch.tensor): index matrix of shape (batch_size, topk)
        
        Returns:
            responses (list[Dict]): list of responses
                - index_database_version (str): version of the index database
                - relevant (list[str]): list of relevant images, each image is a string of image path, sorted by relevance
                - distances (list[int]): list of distances.
        """
        D, I = inference_output
        logger.info(f"Postprocess: I: {I}")
        logger.info(f"Postprocess: D: {D}")
        img_paths: list[list[int]] = [[self.remap_index_to_img_path_dict[str(idx)] for idx in idxs] for idxs in I]
        responses: list[dict] = [{
            "index_database_version": self.setup_config["index_database_version"],
            "relevant": img_path,
            "distances": dists.tolist()
        } for img_path, dists in zip(img_paths, D)]
        logger.info(f"Postprocess: responses: {responses}")
        responses: list[dict] = self.merge_images_with_same_product_id(responses)
        logger.info(f"Responses after merged: {responses}")
        return responses

    
    @staticmethod
    def merge_images_with_same_product_id(responses: list[dict[str, list]]) -> list[dict[str, list]]:
        """
        Merge images with the same product id, uses the first image of each product

        Args:
            responses (list[Dict]): list of responses
        
        Returns:
            responses (list[Dict]): list of responses
        """
        out_responses: list[dict[str, list]] = []
        response: dict[str, list]
        for response in responses:
            out_response: dict[str, list] = {
                "index_database_version": response["index_database_version"],  # type: ignore
                "relevant": [],
                "distances": []
            }
            img_path_to_dist: dict[str, int] = {}
            for img_path, dist in zip(response["relevant"], response["distances"]):
                product = img_path.split("_crop")[0]
                if product not in img_path_to_dist:
                    img_path_to_dist[product] = dist
                else:
                    if dist < img_path_to_dist[product]:
                        raise Exception("The distances are not sorted!")
                    else:
                        continue
                
            for img_path, dist in img_path_to_dist.items():
                out_response["relevant"].append(img_path)
                out_response["distances"].append(dist)
            
            out_responses.append(out_response)

        return out_responses

    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out
