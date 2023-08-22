import torch
import base64
import numpy as np
from ts.torch_handler.vision_handler import VisionHandler
import zipfile
import os
import json
import logging
import cv2
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn as nn
from pymilvus import connections, Collection
import timm
from PIL import Image
import io

logger = logging.getLogger(__name__)

class DeepHashingHandler(VisionHandler):

    @staticmethod
    def data_uri_to_cv2_img(uri: str) -> np.ndarray:
        # https://stackoverflow.com/a/42538142/11806050
        encoded_data = uri.split(',')[-1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def base64_to_PIL_img(uri: str) -> np.ndarray:
        imgdata = base64.b64decode(uri.split(',')[-1])
        img = Image.open(io.BytesIO(imgdata))
        return img

    def __init__(self):
        super(DeepHashingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        yolo_model_path = os.path.join(model_dir, serialized_file)
        model_path = model_dir

        self.device = torch.device( "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu")
        logging.info(f"Using device: {self.device}")

        logger.info(f"Loading model from {model_path}")

        model = timm.create_model(
            'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = model.eval()
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        model.to(self.device)
        self.model = model
        self.transforms = transforms
        logger.info(f'Model loaded successfully from {model_path}: {self.model}')

        logger.info(f"Loading YOLOv8 model from {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        logger.info(f"Loaded YOLOv8 model: {self.yolo_model}")

        # Load the index
        logger.info(f"Loading index ...")
        host = os.environ.get("MILVUS_HOST", "localhost")
        port = os.environ.get("MILVUS_PORT", "19530")
        logger.info(f"Connecting to Milvus: {host}:{port}")
        connections.connect(host=host, port=port)
        logger.info(f"Connected to Milvus: {host}:{port}")
        self.collection = Collection('efiss_image_search')
        logger.info(f"Loading collection: {self.collection.name}")
        self.collection.load()
        logger.info(f"Loaded collection to memory: {self.collection.name}")

        self.search_param = {
            "metric_type": "L2",
            "ignore_growing": False,
            # "params": {"nprobe": 16}
        }

    def search(self, query_embedding: np.ndarray, topk: int) -> tuple[list[list[str]], list[list[float]]]:
        """
        Search the index for the topk most similar images from Milvus database

        Args:
            query_embedding (np.ndarray): the query embedding, batched (1, 768)
            topk (int): the number of results to return
            
        Returns:
            list[list[str]]: the list of image paths
            list[list[float]]: the list of distances
        """
        result = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            # expr=None,
            param={
                "metric_type": "L2", 
                # "offset": 5, 
                # "ignore_growing": False, 
                "params": {}
                # "params": {"nprobe": 10}
            },
            limit=topk,
            output_fields=['path'],
            consistency_level="Strong"
        )
        logger.info(f"result: {result}")

        return [list(row.ids) for row in result], [list(row.distances) for row in result]


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
    
    @staticmethod
    def diversify(image_paths: list[list[str]], distances: list[list[float]], diversity: int) -> tuple[list[list[str]], list[list[float]]]:
        """
        Diversify the results by removing similar images
        
        Args:
            image_paths (list[str]): the list of image paths
            distances (list[float]): the list of distances
            diversity (int): the number of incremental steps to take for pointer
        
        Returns:
            list[str]: the diversified list of image paths
            list[float]: the diversified list of distances
        """
        out_image_paths: list[list[str]] = []
        out_distances: list[list[float]] = []
        
        for image_path, distance in zip(image_paths, distances):
            _out_image_paths: list[str] = []
            _out_distances: list[float] = []

            for i in range(0, len(image_path), diversity):
                _out_image_paths.append(image_path[i])
                _out_distances.append(distance[i])
            
            out_image_paths.append(_out_image_paths)
            out_distances.append(_out_distances)
        
        return out_image_paths, out_distances


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
            diversity: int = req.get("diversity", 1)

            image: np.ndarray = self.data_uri_to_cv2_img(image)
            _images: list[np.ndarray] = self.crop_image(image)

            if len(_images) == 0:
                logger.info("Falling back to the original image")
                _images.append(image)

            # NOTE: for now, we only use the first cropped image
            _cropped_image = _images[0]

            cropped_image: np.ndarray = cv2.cvtColor(_cropped_image, cv2.COLOR_BGR2RGB)

            images.append(cropped_image)
            topk_batch.append(topk)

        assert len(images) == 1, "Currently, we only support one image at a time, edit config.properties to change this behavior"
        
        return images, _cropped_image, topk_batch, diversity

    def inference(self, batch):
        """
        The Inference Function receives the pre-processed input in form of Tensor
        Args:
            batch (torch.tensor): list of images, topk_batch
                - images: torch.tensor of shape (batch_size, 3, image_size, image_size)
                - cropped_image: np.ndarray of shape (3, image_size, image_size)
                - topk_batch (list[int]): list of topk for each image in the batch
        Returns:
            D (torch.tensor): distance matrix of shape (batch_size, topk)
            I (torch.tensor): index matrix of shape (batch_size, topk)
        """
        img_tensor, cropped_image, topk_batch, diversity = batch
        img_tensor: np.ndarray = img_tensor[0]
        topk: int = topk_batch[0]
        logger.info(f"img_tensor.shape: {img_tensor.shape}")
        logger.info(f"topk_batch: {topk_batch}")

        inputs = self.processor(images=img_tensor, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            features: torch.Tensor = self.model(**inputs).logits
        logging.info(f"Features shape: {features.shape}")  # (1, 768)
        logging.info("Finish computing features")

        if len(set(topk_batch)) == 1:
            # all topk are the same, we can use batch search
            images_paths, distances = self.search(features.cpu().numpy(), topk * 2 * diversity)
            # TODO: topk * 4 to ensure there are too few images after filtered by product
        else:
            raise NotImplementedError("Currently, we only support the case where all topk are the same")

        if diversity > 1:
            logger.info(f"Before diversification: {len(images_paths)}: {images_paths}")
            images_paths, distances = self.diversify(images_paths, distances, diversity)
            logger.info(f"After diversification: {len(images_paths)}: {images_paths}")

        return images_paths, distances, cropped_image
    
    def postprocess(self, inference_output):
        """
        The post-process function receives the return value of the inference function.
        It performs post-processing on the raw output to convert it into a format that is
        easy for the user to understand.

        Args:
            inference_output (torch.tensor): list of images, topk_batch
                - images_paths (list[list[str]]): list of list of image paths
                - distances (list[list[float]]): list of list of distances
                - cropped_image (np.ndarray): the cropped image
        
        Returns:
            responses (list[Dict]): list of responses
                - relevant (list[str]): list of relevant images, each image is a string of image path, sorted by relevance
                - distances (list[int]): list of distances.
                - cropped_image (str): base64 encoded image
        """
        images_paths, distances, cropped_image = inference_output
        logger.info(f"Postprocess images_paths: {images_paths}")
        logger.info(f"Postprocess distances: {distances}")
        responses: list[dict] = [{
            "relevant": img_path,
            "distances": dists,
            "cropped_image": _cropped_image
        } for img_path, dists, _cropped_image in zip(images_paths, distances, [cropped_image])]
        logger.info(f"Postprocess: responses: {responses}")
        responses: list[dict] = self.merge_images_with_same_product_id(responses)
        logger.info(f"Responses after merged: {[{'relevant': response['relevant'], 'distances': response['distances'], 'cropped_image': response['cropped_image'][:20]} for response in responses]}")
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
                "relevant": [],
                "distances": [],
                "cropped_image": base64.b64encode(cv2.imencode('.jpg', response["cropped_image"])[1]).decode('utf-8')
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
