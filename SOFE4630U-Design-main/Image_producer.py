from google.cloud import pubsub_v1  # pip install google-cloud-pubsub
import glob  # for searching for JSON file
import base64
import os


service_account_files = glob.glob("*.json")
if not service_account_files:
    raise FileNotFoundError("No service account JSON file found in the current directory.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_files[0]

# Set the Google Cloud project ID and topic name
project_id = "sturdy-mode-448714-s4"
topic_name = "IMAGERECORDS"  # Change if needed

# Create a Pub/Sub publisher with ordering enabled
publisher_options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
publisher = pubsub_v1.PublisherClient(publisher_options=publisher_options)
topic_path = publisher.topic_path(project_id, topic_name)

print(f"Publishing messages with ordering keys to {topic_path}.")

image_folder = "Dataset_Occluded_Pedestrian"
image_files = glob.glob(os.path.join(image_folder, "*.png"))  

if not image_files:
    print("No image files found in the specified folder.")
    exit()


for image_path in image_files:
    try:
        with open(image_path, "rb") as f:
            
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        
        key = os.path.basename(image_path)  

       
        future = publisher.publish(topic_path, encoded_image.encode("utf-8"), ordering_key=key)
        future.result()  
        print(f"Successfully published {key}.")
    
    except Exception as e:
        print(f"Failed to publish {key}: {e}")

print("All messages have been published.")