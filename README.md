There are 2 APIs: query by image and query by text

indices are IDs of images in the database: 0, 1, 2, 3, ...

## Query by image

the input is user's uploaded image and the output is a JSON list of image paths or indices

API: /api/image

Input: multipart/form-data

```
files=user's uploaded image(s)
option={
    "batch_query": false,           // if true, the input containing multiple images
    "return_indices": true,         // if true, return indices instead of image paths
    "top_k": 10,                    // number of images to return
    "algorithm": "h2",           // algorithm to use: m1, m2, m3, h1, h2, h3, t1, t2, t3
}
```

Output:

```
{
    "index_database_version": "1.2.0", // version of the index database
    "batch_query": false,
    "indices": [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
}
```

or

```
{
    "index_database_version": "1.2.0",       // version of the index database
    "batch_query": true,
    "indices": [
        [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],  // results indices for the first image
        [299, 42, 44, 37, 0, 1, 2, 3, 4, 5], // results indices for the second image
        [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
    ]
}
```

### Bash client

```bash
curl -F "option=<option.json" -F "files=@query-image.jpg" http://localhost:8000/api/image
```

or

```bash
curl -F "option=<option.json" -F "files=@query-image1.jpg" -F "files=@query-image2.jpg" -F "files=@query-image3.jpg" http://localhost:8000/api/image
```

### NodeJS client

```javascript
option = {
    batch_query: false,
    return_indices: true,
    top_k: 10,
    algorithm: 'h2',
};
file1 = fs.createReadStream('query-image1.jpg');
file2 = fs.createReadStream('query-image2.jpg');

const formData = new FormData();
formData.append('option', new Blob([JSON.stringify(option)], {type: 'application/json'}));
formData.append('files', file1);
formData.append('files', file2);
fetch('http://localhost:8000/api/image', {
    method: 'POST',
    headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
    },
    body: formData,
}).then(response => response.json()).then(data => {
    console.log(data);
});
```

### Javascript (Browser) client

```javascript
option = {
    batch_query: false,
    return_indices: true,
    top_k: 10,
    algorithm: 'h2',
};
// <input type="file" id="query-image" multiple />
files = document.getElementById('query-image').files;
const formData = new FormData();
formData.append('option', new Blob([JSON.stringify(option)], {type: 'application/json'}));
for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
}
fetch('http://localhost:8000/api/image', {
    method: 'POST',
    headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
    },
    body: formData,
}).then(response => response.json()).then(data => {
    console.log(data);
});
```

## Query by text

the input is a text query and the output is a JSON list of image paths or indices

API: /api/text

Input: application/json

```
{
    batch_query: false,
    return_indices: true,
    top_k: 10, // number of images to return
    query: 'a text query', // text query
    algorithm: 'h2',
}
```

or

```
{
    batch_query: true,
    return_indices: true,
    top_k: 10, // number of images to return
    query: [
        'a text query', // first query
        'another text query', // second query
    ],
    algorithm: 'h2',
}
```

Output:

```
{
    index_database_version: '1.2.0', // version of the index database
    batch_query: false,
    indices: [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],
}
```

or

```
{
    index_database_version: '1.2.0', // version of the index database
    batch_query: true,
    indices: [
        [69, 42, 13, 37, 0, 1, 2, 3, 4, 5],  // results indices for the first query
        [299, 42, 44, 37, 0, 1, 2, 3, 4, 5], // results indices for the second query
    ]
}
```

### Admin API

Reindex the database

API: /api/reindex

Input: application/json

```
{
    new_index_database_version: '1.2.0', // version of the new index database
}
```

Output:

```
{
    result: 'success',
    previous_index_database_version: '1.1.0', // version of the previous index database
    index_database_version: '1.2.0', // version of the new index database
    timestamp: '2020-05-02 12:00:00',
}
```
