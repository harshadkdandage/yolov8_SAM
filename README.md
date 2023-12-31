YOLOv8 with Segment Anything Model

This script captures bounding boxes of detected objects using YOLOv8 and the boxe information is used to segment the objects. 

Install ultralytics and segment_anything: 

pip install ultralytics 

pip install segment_anything

Download weights of pre-trained SAM model:  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Original Image: 

![Cat1](https://github.com/harshadkdandage/yolov8_SAM/assets/47813538/f260f0ab-eef8-464a-a41e-5c84471a769a)

Output: Bounding box data extracted from YOLOv8 detection and segmentation performed within the bounding box.

![image](https://github.com/harshadkdandage/yolov8_SAM/assets/47813538/cd9cc4b6-53e5-4757-918b-af8ec7c4b227)
![image](https://github.com/harshadkdandage/yolov8_SAM/assets/47813538/2c04c949-6cc0-41fd-9f8e-2ca1b45b7b4d)
![image](https://github.com/harshadkdandage/yolov8_SAM/assets/47813538/195cc86a-745b-4c87-8fab-5746053da03e)
![image](https://github.com/harshadkdandage/yolov8_SAM/assets/47813538/688dfafa-bec9-4ea7-87ca-0ca75f8c8604)

This script can be optimized to visualize the bounding box with segmented objects and saved it on a single image with a prediction score as shown in the example below. Individual results are shown in this example for better understanding. 

![cat1_result](https://github.com/harshadkdandage/yolov8_SAM/assets/47813538/c513d79d-b276-48b9-995e-51ce81c2face)
