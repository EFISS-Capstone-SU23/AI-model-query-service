import csv
from glob import glob
from pathlib import Path
from statistics import mean

EMBEDDINGS_PATH = 'embeddings.pt'
DATA_PATH = 'database_info_shopee.txt'

# from towhee import pipe, ops, DataCollection
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name='path', dtype=DataType.VARCHAR, description='path to image', max_length=500, 
                    is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='efiss-image-search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': 'L2',
        'index_type': 'FLAT',
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    return collection
connections.connect(host='192.168.99.2', port='19530')
# connections.connect(host='34.87.182.49', port='19530')
collection = create_milvus_collection('efiss_image_search', 768)
import torch

embeddings = torch.load(EMBEDDINGS_PATH)
import os
import sys
import pandas as pd

df = []
with open(DATA_PATH, 'r') as f:
    for line in f.readlines():
        df.append(line.strip())

df = pd.DataFrame(df)
df
df[0] = df[0].str.replace('data/shopee_crop_yolo/', '')
# _df = df.iloc[:1000]
# _embeddings = embeddings[:1000, :]
# _df.shape, _embeddings.shape
# collection.insert(data={
#     "path": _df[0].values.tolist(),
#     "embedding": _embeddings.cpu().numpy()
# })
# collection.insert(data=[
#     _df[0].values.tolist()[:10],
#     _embeddings.cpu().numpy()[:10, :]
# ])
from tqdm import tqdm, trange
chunk_size = 1000

# milvus insert chunk by chunk
for i in trange(0, len(df), chunk_size):
    collection.insert(data=[
        df[0].iloc[i:i+chunk_size].values.tolist(),
        embeddings[i:i+chunk_size].cpu().numpy()
    ])
collection = Collection("efiss_image_search")
collection.load()
print(collection.num_entities)
# # len(collection.query(expr='path != "milvus.ipynb"'))
# result = collection.search(
#     data=[embeddings[10].cpu().numpy()],
#     anns_field="embedding",
#     # expr=None,
#     param={
#         "metric_type": "L2", 
#         # "offset": 5, 
#         # "ignore_growing": False, 
#         "params": {}
#         # "params": {"nprobe": 10}
#     },
#     limit=10000,
#     output_fields=['path'],
#     # consistency_level="Strong"
# )
# len(result[0].distances)
# type(result[0].ids[0])

# len(result[0].distances)
