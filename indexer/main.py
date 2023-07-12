import argparse
import sys
from utils.logger import setup_logging
from indexer import Indexer
from pprint import pp, pprint
import logging


def main(args):
    new_index_database_version = args.new_index_database_version
    logging.info(f"Begin indexing new_index_database_version: {new_index_database_version}")
    with open(args.database, "r") as f:
        database = f.read().splitlines()
    model_path = args.model_path
    index_mode = args.index_mode
    image_size = args.image_size
    metric_model_path = args.metric_model_path

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
        metric_model_path=metric_model_path,
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
    parser.add_argument('--model_path', help="model_path", type=str, default='torchscripts_models/relahash_tf_efficientnetv2_b3.pt')
    parser.add_argument('--metric_model_path', help="model_path", type=str, default=None)
    parser.add_argument('--dump_index_path', help="path to dump index", type=str, default='index')
    parser.add_argument('--device', help="device", type=str, default='cpu')
    parser.add_argument('--num_workers', help="num_workers", type=int, default=4)
    parser.add_argument('--batch_size', help="batch_size", type=int, default=64)
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