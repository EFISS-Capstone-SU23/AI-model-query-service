import base64
import io
import os
import time
from zipfile import ZipFile
import json

import numpy as np
from flask import Flask, render_template, request, flash, make_response, send_from_directory, jsonify
from utils.datasets import DeepHashingDataset
from utils.indexer import Indexer

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'webp', 'tif', 'jfif'}

def allowed_format(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_web_app(dump_index_path, device='cpu'):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = 'my_secret_key'
        
    @app.route('/api/reindex', methods=['POST'])
    def api_reindex():
        """
        API: /reindex
        {
            "new_index_database_version": "1.2.0",
            "model_path": "wop/abc_xyz/"
        }
        file: database_info.txt
        data/abc.com/69_abc_com.jpg
        data/abc.com/42_abc_com.jpg
        """
        option = json.loads(request.form.get('option'))
        new_index_database_version = option['new_index_database_version']
        print('Begin indexing new_index_database_version: ', new_index_database_version)
        database_info = request.files.getlist('files')[0]
        # read file into a list
        database_info = database_info.read().decode('utf-8').splitlines()
        # ["data/abc.com/69_abc_com.jpg", "data/abc.com/42_abc_com.jpg"]
        model_path = option['model_path']
        index_mode = option.get('mode', 'default')

        results = Indexer.create_index(
            model=model_path,
            new_index_database_version=new_index_database_version,
            database=database_info,
            index_mode=index_mode,
            dump_index_path=dump_index_path
        )
        
        # results = dict(
        #     result='success',
        #     previous_index_database_version='1.1.0',
        #     index_database_version='1.2.0',
        #     timestamp='2020-05-02 12:00:00',
        #     elapsed_time=100, # seconds
        # )
        return jsonify(results)

    return app