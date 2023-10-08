import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from PIL import Image
import traceback
import os
import io

class GenerateThumbnail(beam.DoFn):
    def __init__(self, bucket_name="efiss"):
        self.bucket_name = bucket_name
        self.new_height = 150
        self.gcs = None
    
    def setup(self):
        # self.gcs = beam.io.gcp.gcsio.GcsIO()
        # self.gcs2  = beam.io.gcp.gcsio.GcsIO()
        self.io = io
        self.Image = Image
        
    def process(self, element):
        print(f"Processing {element}...")
        # input_path = f"gs://{self.bucket_name}/{element}"
        input_path = f"/media/thaiminhpv/DataStorage/EFISS/data/{element}"
        # print(input_path)
        output_path = input_path.replace("product_images", "thumbnail")
        # print(output_path)
        
        try:
            if os.path.exists(output_path):
                print(f"Thumbnail already exists for {element}.")
                return [output_path]

            # Load the image
            with open(input_path, "rb") as f:
                image_bytes = f.read()

            # Generate thumbnail
            img = self.Image.open(self.io.BytesIO(image_bytes))
            original_width, original_height = img.size
            new_width = int(original_width * (self.new_height / original_height))
            img.thumbnail((new_width, self.new_height))
            
            # Save the thumbnail to GCS
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                # mkdir -p parent_dir
                img.save(f, format=img.format)

            print(f"Done processing {element}.")
            return [output_path]
        except Exception as e:
            print(f"Error processing {element}: {e}")
            print(traceback.format_exc())
            return [f"Error processing {element}: {e}"]


if __name__ == "__main__":
    bucket_name = 'efiss'
    pipeline_options = PipelineOptions(
        # runner="DataflowRunner",
        # project="efiss-duong",
        # # project='efiss-393918',
        # region="us-central1",
        # # region="asia-southeast1",
        # # worker_zone="asia-southeast1-b",
        # machine_type="n1-standard-2",
        # temp_location="gs://efiss-tmp-us/temp",
        # requirements_file='requirements.txt',
        # autoscaling_algorithm='NONE',
        # num_workers=2,
        # max_num_workers=2,
        # number_of_worker_harness_threads=20,
        direct_num_workers=30,
        direct_running_mode='multi_threading',
        save_main_session=True,
    )
    with beam.Pipeline(options=pipeline_options) as p:
        files = (
            p
            # | 'List Files' >> beam.io.ReadFromText(f"gs://{bucket_name}/queue/to_be_thumbnail3.txt")
            | 'List Files' >> beam.io.ReadFromText(f"_.txt")
            | 'Process Files' >> beam.ParDo(GenerateThumbnail())
        )
