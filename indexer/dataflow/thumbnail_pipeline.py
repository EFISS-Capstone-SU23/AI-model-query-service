import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from PIL import Image
import io

from google.cloud import storage

# # # def list_blobs(bucket_name):
# # #     # Initialize a client
# # #     client = storage.Client()

# # #     # Get the bucket
# # #     bucket = client.get_bucket(bucket_name)

# # #     # List blobs in the bucket with the given prefix
# # #     prefix = 'data/product_images/'
# # #     blobs = bucket.list_blobs(prefix=prefix)

# # #     return [blob.name for blob in blobs]

# # # bucket_name = "efiss"  # Replace with your actual bucket name
# # files_list = list_blobs(bucket_name)
# # print(len(files_list))
# with open('files_list_efiss.txt', 'w') as f:
#     for item in files_list:
#         f.write("%s\n" % item)
# with open('files_list_efiss.txt', 'r') as f:
#     files_list = f.read().splitlines()

class GenerateThumbnail(beam.DoFn):
    def __init__(self, bucket_name="efiss"):
        self.bucket_name = bucket_name
        self.gcs = beam.io.gcsio.GcsIO()
        self.new_height = 150
        
    def process(self, element):
        print(f"Processing {element}...")
        input_path = f"gs://{self.bucket_name}/{element}"
        output_path = input_path.replace("product_images", "thumbnail")
        
        # Load the image from GCS
        with self.gcs.open(input_path, "rb") as f:
            image_bytes = f.read()

        # Generate thumbnail
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size
        new_width = int(original_width * (self.new_height / original_height))
        img.thumbnail((new_width, self.new_height))
        
        # Save the thumbnail to GCS
        with self.gcs.open(output_path, "wb") as f:
            img.save(f, format=img.format)

        print(f"Done processing {element}.")
        return [output_path]


def run_pipeline(bucket_name):
    pipeline_options = PipelineOptions(
        runner="DataflowRunner",
        project="efiss-duong",
        # project='efiss-393918',
        # region="us-central1",
        region="asia-southeast1",
        worker_zone="asia-southeast1-b",
        machine_type="n1-standard-2",
        temp_location="gs://efiss-temp/temp",
        requirements_file='requirements.txt',
        autoscaling_algorithm='NONE',
        num_workers=1,
        number_of_worker_harness_threads=24,
        save_main_session=True,
    )
    with beam.Pipeline(options=pipeline_options) as p:
        files = (
            p
            | 'List Files' >> beam.io.ReadFromText(f"gs://{bucket_name}/data/files_list_efiss.txt")
            | 'Process Files' >> beam.ParDo(GenerateThumbnail(bucket_name))
        )

if __name__ == '__main__':
    bucket_name = 'efiss'
    run_pipeline(bucket_name)
