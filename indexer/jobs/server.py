import os
import pickle
from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI()

# Define the directory where you want to save the uploaded payload
upload_dir = "uploads"

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

@app.post("/upload/")
async def upload_file(file: UploadFile):
    try:
        # Read the uploaded file
        payload_bytes = await file.read()

        # Deserialize the payload using pickle
        payload = pickle.loads(payload_bytes)

        # Save the payload to disk
        shard_id = payload["shard_id"]
        shard_path = os.path.join(upload_dir, f"shard_{shard_id}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(payload, f)

        return {"message": "Payload received and saved successfully"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
