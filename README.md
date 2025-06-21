How to use:
- Linux:
  + git clone https://github.com/F4tt/Dispatch-Monitoring-System.git
  + cd EATLAB
  + add video to folder, change video path if using another video.
  ![Ảnh chụp màn hình 2025-06-17 161452](https://github.com/user-attachments/assets/ae8596a8-7651-4bee-8589-88156614c3a3)
  + docker-compose up
- Windows:
  + git clone https://github.com/F4tt/Dispatch-Monitoring-System.git
  + cd EATLAB
  + add video to folder, change video path if using another video.
  + python main.py

Features:
- Press "r" to detect and classify objects within the ROI (Region of Interest).
- Track objects and assign IDs.
- Press "f" to open a Feedback GUI that allows relabeling of misclassified objects and saving them to the feedback_data folder.
![Ảnh chụp màn hình 2025-06-16 192100](https://github.com/user-attachments/assets/5027ee10-f0ea-43ac-b342-d7f55778b81f)
![Ảnh chụp màn hình 2025-06-16 192352](https://github.com/user-attachments/assets/0814fbd5-8dac-48cd-a27d-da57a4196b39)
- Automatic retraining based on feedback data. The retrained model is saved if its performance surpasses the current one.
 ![image](https://github.com/user-attachments/assets/e2a65f66-49b8-4d8e-909f-2d381c2c6b69)
- Model switching: Press "n" to use the newer model, and press "p" to revert to the previous model.
 
  
Choices and Justifications:
- Kaggle Notebook: I use Kaggle for training machine learning models because it offers free and powerful GPU resources, while my laptop lacks a capable GPU.
- YOLOv5n / YOLOv8n: I chose YOLO because it offers excellent accuracy, is actively updated, and is lightweight—especially with the YOLOn/YOLOs versions—making it suitable for camera or edge devices.
- ResNet18 / MobileNetV2 / EfficientNet_B0: These are the classification models I experimented with. ResNet18 represents a moderately-sized model, while MobileNetV2 and EfficientNet_B0 are more lightweight. I aimed to compare speed and accuracy to determine the best fit for a resource-constrained environment.
- ROI (Region of Interest): The detection dataset clearly focuses on a specific region. Detecting over the entire frame wastes performance and may lower accuracy. Thus, I used ROI to ensure the model only focuses on the relevant area, addressing these concerns effectively.
- System Pipeline:
ROI → crop → detect (no dish/tray classification) → crop → classify (6 labels).
I considered training a direct multi-class model (e.g., YOLO for all 6 labels), but the dataset was not labeled appropriately. I also thought about having YOLO classify dish/tray and then building two separate classifiers for each group. However, the original pipeline worked quite well, so I concluded that other approaches might not yield significant improvements.
- Multithreading & Frame Skipping: To reduce system load, I separated tasks into threads and used FRAME_SKIP = 3 to process one out of every three frames. This still ensures enough information is captured from the video while easing the computational burden.


Workflow:
1. Problem Analysis
- Designed a camera-based surveillance system → requiring lightweight yet accurate models.
- Detection dataset is region-specific → Solution: use ROI.
- Classification dataset:
  + The “dish_empty” class is overrepresented.
  + Other labels are balanced.
- If this causes model bias, I plan to reduce its frequency or apply data augmentation to other classes to improve performance.
- Manual filtering of noisy data is ineffective → Instead, inspect wrongly predicted images during evaluation to identify and fix noisy samples.
2. Model Training
2.1. Detection Models
- The dataset is already split into train/val sets and is ready for use with YOLOv5n/YOLOv8n, so no preprocessing was done initially.
- Finetuned both YOLOv5n and YOLOv8n.
- Applied early stopping and saved the best model during training.
- Each model outputs detailed training and validation metrics: box_loss, cls_loss, dfl_loss, Precision, Recall, mAP50, and mAP50-95.
2.2. Classification Models
- The dataset is split into 80% training and 20% validation for each label.
- Finetuned 3 models: ResNet18, MobileNetV2, and EfficientNet_B0.
- Replaced the final fully connected layers.
- Evaluated on the validation set after each epoch.
- Used early stopping and saved the best model.
- Evaluated models using multiple metrics: loss, accuracy, precision, recall, and f1-score.

  2.3. Training Results
  - Yolov5n:
  ![image](https://github.com/user-attachments/assets/e4e5d42e-7068-4608-97d9-22b8bb20b65e)
  - Yolov8n:
  ![image](https://github.com/user-attachments/assets/e16a0875-607e-4388-acc6-c45f714632f3)
  - Resnet18:
  ![image](https://github.com/user-attachments/assets/01f12934-198e-4a57-8859-ee792ccc56d9)
  - Mobilenetv2:
  ![image](https://github.com/user-attachments/assets/371db0aa-1eb9-4b39-8144-cb4a55333844)
  - Efficientnet_b0:
  ![image](https://github.com/user-attachments/assets/f2257029-67ee-4c7b-882d-0ba97f1fb6e0)

  ![image](https://github.com/user-attachments/assets/6d91f6d0-9021-46e9-ab1e-0e0a792046dc)

3. Inference.
- Pipeline: ROI → crop → detect (no dish/tray classification) → crop → classify (6 labels)
- Retraining Process: Crop objects from frame → open feedback gui → relabel misclassified ones → save to feedback_data → automatically retrain once the feedback for each label grows by +20 images compared to last retraining → save the new model if it outperforms the previous one; otherwise, discard it.
- Model Switching: Swap the current classification model with the latest retrained one or roll back to a previous version as needed.
- Why ROI: As mentioned, the detection dataset focuses on a particular area. Detecting across the entire frame consumes more resources and may reduce system accuracy. ROI allows the model to focus on the relevant zone, resolving these issues efficiently.

4. Optimization Techniques
- Multithreading: Separate threads for detection, classification, and frame display
- Frame Skipping: Process every 3rd frame with FRAME_SKIP = 3
- Mixed Precision: Use half=True (float16) if the GPU supports it
- Batch Processing: Apply batched inference to accelerate classification

5. Deployment - Docker
