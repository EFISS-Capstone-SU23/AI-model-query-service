import json
from flask import (
    Flask,
    request,
    jsonify,
)
from utils.indexer import Indexer
import logging


def get_web_app(configs):
    indexer_service = Indexer(configs)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "my_secret_key"

    @app.route("/api/reindex", methods=["POST"])
    def api_reindex():
        """
        API: /reindex
        {
            "new_index_database_version": "1.2.0",
            "model_path": "torchscripts_models/relahash_tf_efficientnetv2_b3.pt",
        }
        file: database_info.txt
        data/abc.com/69_abc_com.jpg
        data/abc.com/42_abc_com.jpg
        """
        option = json.loads(request.form.get("option"))
        new_index_database_version = option["new_index_database_version"]
        logging.info(f"Begin indexing new_index_database_version: {new_index_database_version}")
        database_info = request.files.getlist("files")[0]
        # read file into a list
        database_info = database_info.read().decode("utf-8").splitlines()
        # ["data/abc.com/69_abc_com.jpg", "data/abc.com/42_abc_com.jpg"]
        model_path = option["model_path"]
        index_mode = option.get("mode", "default")
        image_size = option.get("image_size", None)
        if image_size is None:
            if "tf_efficientnetv2_b3" in model_path:
                image_size = 300
            elif "tf_efficientnet_b7_ns" in model_path:
                image_size = 600
            elif "mobilenetv3small" in model_path:
                image_size = 224
            else:
                raise ValueError("image_size is not specified")
        logging.info(f"image_size: {image_size}")

        results = indexer_service.create_index(
            model_path=model_path,
            image_size=image_size,
            new_index_database_version=new_index_database_version,
            database=database_info,
            index_mode=index_mode,
        )

        logging.info(f"Return results: {results}")
        return jsonify(results)

    return app
