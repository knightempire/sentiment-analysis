# Sentiment Analysis Project

### Contributors
- Abinesh T, CB.EN.U4CSE21302
- Kaviswar R, CB.EN.U4CSE21345
- Leonal Robin D, CB.EN.U4CSE21331
- Tricia Ezhilarasi J, CB.EN.U4CSE21265

## Problem Statement

Emotion recognition from facial expressions is challenging due to variations in lighting, angles, and expression intensity within datasets. This research aims to develop a machine learning model that accurately classifies emotions such as happiness, sadness, and anger, despite these challenges. By implementing preprocessing, feature extraction, and robust classification techniques, this study seeks to improve model accuracy and generalizability, ultimately enhancing user experience in interactive applications.

---

## Dataset Information

- **Training Dataset**: Labeled dataset from Kaggle, which includes 902 images representing seven emotions: happiness, sadness, fear, anger, disgust, surprise, and neutrality. 
- **Testing Dataset**: A separate collection of images downloaded from the internet, covering varied lighting conditions, angles, and real-world scenarios to test the model's generalization ability.

- **Dataset Link**: [Kaggle Emotion Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

---

## Methodology and Models

### 1. **Custom CNN Model**

#### Methodology
This custom model leverages convolutional blocks with batch normalization, max-pooling, and dropout for regularization. The architecture progressively extracts high-level features, which improve emotion classification performance.

#### Algorithm
1. **Input Layer**: Grayscale images of shape (48x48x1).
2. **Convolutional Blocks**:
   - Block 1: 64 filters, 3x3 kernel, ReLU activation, batch normalization.
   - Block 2: Same as Block 1, with max-pooling and dropout.
   - Additional blocks with increasing filters (128, 256) follow the same pattern.
3. **Fully Connected Layers**:
   - Dense layer with 128 units, followed by batch normalization and dropout.
4. **Output Layer**: 7 units with softmax activation for multiclass classification.

#### Training and Testing Metrics
- **Accuracy**: 85%
- **Precision**: 0.83
- **Recall**: 0.84
- **F1-Score**: 0.83

### 2. **AlexNet**

#### Methodology
AlexNet, designed for large-scale image classification, is modified here to suit emotion recognition tasks. This model uses ReLU activations and dropout for regularization.

#### Algorithm
1. **Input Layer**: RGB image, 224x224x3.
2. **Convolutional Layers**: Multiple layers with varying filter sizes, starting with 96 filters.
3. **Max Pooling and Dropout**: Applied after each convolutional block to reduce dimensions and avoid overfitting.
4. **Fully Connected Layers**: Includes dropout for regularization and ReLU activation.
5. **Output Layer**: Softmax activation for classification.

#### Training and Testing Metrics
- **Accuracy**: 87%
- **Precision**: 0.85
- **Recall**: 0.86
- **F1-Score**: 0.85

### 3. **DenseNet121**

#### Methodology
DenseNet121 is pre-trained on ImageNet and fine-tuned on our dataset. Dense connections between layers improve gradient flow and model efficiency.

#### Algorithm
1. **Dense Blocks**: Each layer is connected to every other layer in a feed-forward manner.
2. **Transition Layers**: Includes batch normalization, ReLU activation, and 1x1 convolution.
3. **Fully Connected Layers**: Followed by dropout and a dense layer with 512 units.
4. **Output Layer**: 7 units with softmax activation.

#### Training and Testing Metrics
- **Accuracy**: 89%
- **Precision**: 0.88
- **Recall**: 0.88
- **F1-Score**: 0.88

---

## Evaluation Metrics

For each model, we evaluated performance using the following metrics:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Ratio of correctly predicted positive observations to total positive observations.
- **Recall**: Ability to detect all positive instances.
- **F1-Score**: Weighted average of precision and recall, balancing both.

### Confusion Matrix and Report
The confusion matrix and detailed report were generated for each model, offering insights into class-wise performance and helping refine the models.

---

## Explainable AI (XAI) Techniques

To enhance transparency and trust:
- **Grad-CAM**: Highlights important regions in the image for each emotion.
- **LIME**: Local explanations for individual predictions.
- **SHAP Values**: Shows feature contributions to model predictions.

---

## Future Enhancements

Potential improvements include:
1. **Federated Learning**: To improve privacy by training across decentralized devices.
2. **Data Augmentation with Generative AI**: Use GANs to create synthetic emotion data, enhancing model diversity.
3. **Optimization for Real-Time Applications**: Improve processing speeds and reduce latency for real-time emotion detection.

---

## References
1. Gidudu, A., et al. "Classification of Images Using Support Vector Machines."
2. Zhu, F., et al. "Image classification method based on improved KNN algorithm."
3. Bosch, A., et al. "Image Classification using Random Forests and Ferns."

---

### Technology Stack
- **Python**, **TensorFlow**, **scikit-learn**, **Matplotlib**, **Streamlit** for web interface and webcam integration.
