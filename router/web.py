import json
from flask import (
    Flask,
    request,
    jsonify,
)
import logging


def get_web_app(configs):
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
        # run bash script
        ...

    return app
