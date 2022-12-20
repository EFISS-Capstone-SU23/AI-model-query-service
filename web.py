import base64
import io
import os
import time
from zipfile import ZipFile
import json

import numpy as np
from flask import Flask, render_template, request, flash, make_response, send_from_directory, jsonify

# from inference.indexer import Indexer
# from utils.misc import pil_loader

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'webp', 'tif', 'jfif'}


def allowed_format(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_web_app(log_path, device='cpu', top_k=10):
    # indexer = Indexer(log_path, device=device, top_k=top_k)
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = 'my_secret_key'

    """
    There are 2 APIs: query by image and query by text
    indices are IDs of images in the database: 0, 1, 2, 3, ...

    For query by image, the input is user's uploaded image and the output is a JSON list of image paths or indices
    API: /api/image
    Input: multipart/form-data
    file=user's uploaded image
    files=user's uploaded images (if batch_query is true)
    option={
        batch_query: false, // if true, the input containing multiple images
        return_indices: true, // if true, return indices instead of image paths
        top_k: 10, // number of images to return
    }
    Output:
    {
        index_database_version: '1.2.0', // version of the index database
        batch_query: false,
        indices: [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
    }
    or
    {
        index_database_version: '1.2.0', // version of the index database
        batch_query: true,
        indices: [
            [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],  // results indices for the first image
            [299, 42, 44, 37, 0, 1, 2, 3, 4, 5], // results indices for the second image
            [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
        ]
    }

    For query by text, the input is a text query and the output is a JSON list of image paths or indices
    API: /api/text
    Input: application/json
    {
        batch_query: false,
        return_indices: true,
        top_k: 10, // number of images to return
        query: 'a text query', // text query
    }
    or
    {
        batch_query: true,
        return_indices: true,
        top_k: 10, // number of images to return
        query: [
            'a text query', // first query
            'another text query', // second query
        ],
    }
    Output:
    {
        index_database_version: '1.2.0', // version of the index database
        batch_query: false,
        indices: [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
    }
    or
    {
        index_database_version: '1.2.0', // version of the index database
        batch_query: true,
        indices: [
            [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],  // results indices for the first query
            [299, 42, 44, 37, 0, 1, 2, 3, 4, 5], // results indices for the second query
        ]
    }
    
    ### Admin API:
    Reindex the database
    API: /api/reindex
    Input: application/json
    {
        new_index_database_version: '1.2.0', // version of the new index database
    }
    Output:
    {
        result: 'success',
        previous_index_database_version: '1.1.0', // version of the previous index database
        index_database_version: '1.2.0', // version of the new index database
        timestamp: '2020-05-02 12:00:00',
    }
    """
    
    # API
    @app.route('/api/image', methods=['POST'])
    def api_image():
        option = json.loads(request.form.get('option'))
        # do not need to [0] to not be unsqueeze later
        images = request.files.getlist('files')
        print('option ', option)
        print('images ', images)
        ... # process the images and return the results
        results = dict(
            index_database_version='1.2.0',
            batch_query=False,
            indices=[69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
            # or
            # indices=[
            #     [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
            #     [299, 42, 44, 37, 0, 1, 2, 3, 4, 5],
            #     [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
            # ],
        )
        return jsonify(results)
    
    @app.route('/api/text', methods=['POST'])
    def api_text():
        option = request.json['option']
        query = request.json['query']
        print('option ', option)
        print('query ', query)
        ... # process the query and return the results
        results = dict(
            index_database_version='1.2.0',
            batch_query=False,
            indices=[69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
            # or
            # indices=[
            #     [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
            #     [299, 42, 44, 37, 0, 1, 2, 3, 4, 5],
            #     [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
            # ],
        )
        return jsonify(results)
        
    @app.route('/api/reindex', methods=['POST'])
    def api_reindex():
        new_index_database_version = request.json['new_index_database_version']
        print('new_index_database_version ', new_index_database_version)
        ... # reindex the database
        results = dict(
            result='success',
            previous_index_database_version='1.1.0',
            index_database_version='1.2.0',
            timestamp='2020-05-02 12:00:00',
        )
        return jsonify(results)

    return app

// bash client
curl -F "option=<option.json" -F "files=@app.py" -F "files=@web.py" http://localhost:8000

// javascript client
const formData = new FormData();
formData.append('option', new Blob([JSON.stringify(option)], {type: 'application/json'}));
formData.append('files', file1);
formData.append('files', file2);
fetch('http://localhost:8000', {
    method: 'POST',
    headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
    },
    body: formData,
}).then(response => response.json()).then(data => {
    console.log(data);
});