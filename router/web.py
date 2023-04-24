import json
from flask import (
    Flask,
    request,
    jsonify,
)
import logging
import os
import sys
import zipfile
from indexer import main
from indexer.utils.logger import setup_logging
import requests

DATA_LAKE_PATH = "/media/saplab/Data_Win/RSI_Do_An/AnhND/Dynamic-Crawler-Tool/output"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "my_secret_key"
setup_logging('router.log')


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """
    API: /reindex
    {
        "new_index_database_version": "1.5.0",
        "model_path": "torchscripts_models/relahash_tf_efficientnetv2_b3.pt",
    }
    """
    # 1. Create database_info.txt file
    # 2. run script to reindex the database 
    # 3. then create new .mar file using bash script
    # 4. then fetching torchserve management API to update the model with new .mar file
    
    data = request.get_json()
    new_index_database_version = data.get("new_index_database_version", "1.5.0")
    model_path = data.get("model_path", "torchscripts_models/relahash_tf_efficientnetv2_b3.pt")

    # 1. Create database_info.txt file
    logging.info(f"----------- 1. Begin create database_info.txt file -----------")
    os.system(f"./indexer/extract_datalake.sh {DATA_LAKE_PATH}")
    logging.info(f"----------- 1. End create database_info.txt file -----------")


    # 2. run script to reindex the database 
    logging.info(f"----------- 2. Begin reindex the database -----------")
    command = f"""
    python indexer/main.py \
        --database database_info.txt \
        --model_path {model_path} \
        --device cuda:0 \
        --new_index_database_version {new_index_database_version}
    """
    command = ''.join(s for s in command if s != '"').split()[2:]
    main.run(command)
    logging.info(f"----------- 2. End reindex the database -----------")


    # 3. then create new .mar file using bash script
    logging.info(f"----------- 3. Begin create new .mar file -----------")
    os.system(f"./searcher/autoserve.sh relahash-medium-64bits {new_index_database_version} {model_path}")
    logging.info(f"----------- 3. End create new .mar file -----------")

    
    # 4. then fetching torchserve management API to update the model with new .mar file
    logging.info(f"----------- 4. Begin fetching torchserve management API to update the model with new .mar file -----------")
    # requests.post("http://localhost:8081/models/relahash-medium-64bits?initial_workers=1&synchronous=true", files={"model": open(f"searcher/relahash-medium-64bits-{new_index_database_version}.mar", "rb")})
    raise Exception("Not implement yet")
    
