import cv2
from ultralytics import YOLO
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_images(imgs):
    for img in imgs:
        img = img.plot()
        cv2.imshow("Detection", img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        

def object_detection():
    image_files = glob.glob("pedestrian_images/*.*")
    print(os.path.abspath("pedestrian_images"))
    image_files = image_files[:5]
    detected_objects = yolo_model(source="pedestrian_images", show=False, conf=0.4, classes=0)
    # detected_objects = model(source="yolov11/pedestrian_images", show=False, conf=0.4, classes=0)
   

    return detected_objects

def get_depth_map(img):
    if midas_model == "DPT_Large" or midas_model == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

yolo_model = YOLO("yolo11n.pt")

midas_model = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", midas_model)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

results = object_detection()
images = []

output_dir = "output_images/"
os.makedirs(output_dir, exist_ok=True)

for i, result in enumerate(results):

    if len(result.boxes) == 0:
        continue

    image = result.orig_img
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = get_depth_map(rgb_image)

    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        person_depth_map = depth_map[y1:y2, x1:x2]
        median_depth = np.median(person_depth_map)
        print(f"Estimated Depth of Person: {median_depth:.2f} MiDaS Units")
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (60, 179, 113), 2)
        cv2.putText(rgb_image, f"Depth: {median_depth:.2f} MiDaS Units", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 179, 113), 2)


    # Show the original image with detections
    output_path = f'{output_dir}/object_detected{i}.png'
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("YOLO + Depth Estimation")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # plt.show()
    #
    # # Show the depth map
    # plt.imshow(depth_map, cmap="inferno")
    # plt.axis("off")
    # plt.colorbar()
    # plt.title("Depth Map")
    # plt.show()