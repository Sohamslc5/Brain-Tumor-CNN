# Brain Tumor Classification using Convolutional Neural Network and Vertex AI
This Mini Project was to develop CNN Model for Brain Tumor Classification using CNN and understanding the basics of deep learning and to get hands-on experience with Vertex AI.
## Description
The objective of our project was to develop a Convolutional Neural Network (CNN) model for classifying brain tumors from medical images and integrate this model into a web application using Vertex AI for API management and deployment.

This project is created by a group of four 5th Semester students of IIITA and the duration of the project was July - December (2023). The members with their respective roll number are mentioned below:

  - IIB2021032 - Roshan Chaudhary
  - IIB2021040 - Roushan Kumar
  - IIB2021043 - Soham Chauhan
  - IIB2021044 - HimaBindu Jadhav

## Project Breakdown
1. **Data Collection and Preprocessing**: We began by gathering a comprehensive dataset of brain MRI scans, which included images labeled with different types of brain tumors such as gliomas, meningiomas, and pituitary tumors. The dataset also included normal brain scans for control. Data preprocessing involved several critical steps:
    - Normalization: Adjusting the pixel values to a standard scale.
    - Augmentation: Applying transformations such as rotations, zooming, and flipping to increase the diversity of the training data.
    - Segmentation: Isolating regions of interest (tumor regions) to enhance the model's focus.
  
2. **Model Development and Training**: The model development involved constructing a Convolutional Neural Network (CNN) architecture with convolutional, pooling, and fully connected layers for feature extraction and classification. Training utilized supervised learning techniques, optimizing with metrics such as accuracy, precision, recall, and F1-score.

3. **Model Deployment**: Deployment employed Google Cloud's Vertex AI, enabling seamless integration into a web application. The trained model was exported, uploaded to Google Cloud Storage, and deployed as an endpoint with Vertex AI. API key access was managed for real-time predictions, ensuring scalability and reliability for medical professionals using the web interface.
