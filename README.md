# Digi_Analyser
🧠 Handwritten Digit Recognition using Neural Network

A neural-network-based handwritten digit recognition project using the MNIST dataset and TensorFlow/Keras. The project focuses not only on building a neural network but also on understanding how different hyperparameters affect model performance.

📌 Project Overview

Handwritten Digit Recognition is a classification problem in which a machine-learning model identifies a handwritten digit from an image.

In this project, a neural network is trained to classify grayscale images of handwritten digits from 0 to 9 using the MNIST dataset.

Each MNIST image has a resolution of:

28 × 28 pixels


Therefore, each image contains:

28 × 28 = 784 pixels


The images are processed and provided to a neural network, which learns patterns from the pixel values and predicts the corresponding digit.

Basic workflow
MNIST Image
     ↓
Data Preprocessing
     ↓
Normalization
     ↓
Flatten 28 × 28 → 784
     ↓
Neural Network
     ↓
Hidden Layers
     ↓
Output Layer
     ↓
Prediction: 0–9

🎯 Objectives

The main objectives of this project are:

Understand the fundamentals of neural networks.
Understand neurons, weights, biases, and activation functions.
Load and preprocess the MNIST dataset.
Build a neural network using TensorFlow/Keras.
Train the network for handwritten digit classification.
Evaluate the model using accuracy and loss.
Experiment with different neural-network architectures.
Study the effect of:
Number of hidden layers
Number of neurons
Batch size
Learning rate
Number of epochs
Optimizers
Dropout
Compare experimental results and identify a suitable model configuration.
📚 About the MNIST Dataset

MNIST is a widely used dataset for handwritten digit classification.

It contains:

60,000 training images
10,000 testing images
10 classes: digits 0 through 9
Image size: 28 × 28 pixels
Image type: grayscale

Example:

Input Image
     ↓
28 × 28 pixels
     ↓
784 pixel values
     ↓
Neural Network
     ↓
One of 10 classes
     ↓
0, 1, 2, ..., 9

🧠 Neural Network Fundamentals
What is a Neural Network?

A neural network is a mathematical model that learns a mapping between input data and output labels.

For this project:

Input  →  MNIST Image
Output →  Digit 0–9


Mathematically:

𝑓
(
𝑋
;
𝜃
)
→
𝑌

where:

𝑋
 = input image
𝑌
 = target digit
𝜃
 = learnable parameters such as weights and biases
🧩 Network Architecture

The baseline network used in this project is:

Input
784 neurons
     ↓
Dense Layer
128 neurons
ReLU
     ↓
Dense Layer
64 neurons
ReLU
     ↓
Output Layer
10 neurons
Softmax


Therefore, the architecture can be represented as:

784 → 128 → 64 → 10

Why 784 input features?

MNIST images have dimensions:

28
×
28
=
784

The image is flattened before being passed to the fully connected layers.

Why 10 output neurons?

There are ten possible digit classes:

0 1 2 3 4 5 6 7 8 9


Therefore, the output layer contains 10 neurons.

Why 128 and 64 hidden neurons?

These values are hyperparameters, not fixed requirements.

They were selected as a reasonable baseline architecture. Different numbers of neurons can be tested experimentally to determine their effect on accuracy, generalization, and computational cost.

⚙️ How a Neuron Works

A neuron performs a weighted sum followed by an activation function.

𝑧
=
∑
𝑖
=
1
𝑛
𝑤
𝑖
𝑥
𝑖
+
𝑏

Then:

𝑎
=
𝑓
(
𝑧
)

where:

𝑥
𝑖
 = input
𝑤
𝑖
 = weight
𝑏
 = bias
𝑓
 = activation function
𝑎
 = neuron output

In matrix form:

𝐴
=
𝑓
(
𝑊
𝑋
+
𝑏
)

🔥 Activation Functions
ReLU

The hidden layers use the ReLU activation function:

𝑅
𝑒
𝐿
𝑈
(
𝑥
)
=
max
⁡
(
0
,
𝑥
)

Example:

Input:  [-2, -1, 0, 2, 5]

ReLU:   [ 0,  0, 0, 2, 5]


ReLU introduces non-linearity into the network and is widely used in neural networks.

Softmax

The output layer uses Softmax because this is a 10-class classification problem.

Softmax converts output scores into probabilities:

𝑃
𝑖
=
𝑒
𝑧
𝑖
∑
𝑗
𝑒
𝑧
𝑗

For example:

Digit       Probability

0             0.01
1             0.00
2             0.02
3             0.01
4             0.01
5             0.90
6             0.01
7             0.01
8             0.02
9             0.01


The model predicts the class with the highest probability:

Prediction = 5

🔄 Forward Propagation

During forward propagation, the input travels through the network.

For the baseline model:

𝐴
1
=
𝑅
𝑒
𝐿
𝑈
(
𝑊
1
𝑋
+
𝑏
1
)

𝐴
2
=
𝑅
𝑒
𝐿
𝑈
(
𝑊
2
𝐴
1
+
𝑏
2
)

𝑌
^
=
𝑆
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑊
3
𝐴
2
+
𝑏
3
)

The complete process is:

Input Image
     ↓
Flatten
     ↓
784 Features
     ↓
Dense(128)
     ↓
ReLU
     ↓
Dense(64)
     ↓
ReLU
     ↓
Dense(10)
     ↓
Softmax
     ↓
10 Probabilities
     ↓
Predicted Digit

📉 Loss Function

The loss function measures how different the prediction is from the correct label.

This project uses:

Sparse Categorical Cross-Entropy


For a single example:

𝐿
=
−
log
⁡
(
𝑦
^
𝑐
𝑜
𝑟
𝑟
𝑒
𝑐
𝑡
)

If the model assigns a high probability to the correct class, the loss is small.

If it assigns a low probability to the correct class, the loss is large.

The objective of training is therefore:

min
⁡
𝜃
𝐿
(
𝜃
)

🔙 Backpropagation

After calculating the loss, the neural network calculates gradients using backpropagation.

The training process is:

Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Calculate Loss
  ↓
Backpropagation
  ↓
Calculate Gradients
  ↓
Optimizer
  ↓
Update Weights
  ↓
Next Batch


Backpropagation uses the chain rule of calculus to determine how each weight and bias contributed to the error.

🚀 Optimizer

The optimizer updates the model parameters using the calculated gradients.

This project uses Adam as the baseline optimizer.

A simplified gradient-descent update is:

𝑊
𝑛
𝑒
𝑤
=
𝑊
𝑜
𝑙
𝑑
−
𝜂
∇
𝑊
𝐿

where:

𝑊
 = model weight
𝐿
 = loss
𝜂
 = learning rate

Adam improves upon simple gradient descent by adapting updates using information from previous gradients.

🎛️ Hyperparameters

Hyperparameters are settings selected by the developer rather than directly learned by the model.

The major hyperparameters investigated in this project are:

Hyperparameter	Description
Number of layers	Controls network depth
Number of neurons	Controls model capacity
Learning rate	Controls update size
Batch size	Number of samples used per update
Epochs	Number of complete passes through training data
Optimizer	Determines how parameters are updated
Activation function	Determines the nonlinear transformation
Dropout	Regularization technique
🔬 Experiments

A major part of this project is studying how different hyperparameters affect performance.

Experiment 1 — Number of Hidden Layers

Possible architectures:

Model A: 784 → 128 → 10

Model B: 784 → 128 → 64 → 10

Model C: 784 → 256 → 128 → 64 → 10


The objective is to observe how increasing network depth affects:

Accuracy
Loss
Training time
Generalization
Experiment 2 — Number of Neurons

The number of neurons can be varied:

32
64
128
256


For example:

784 → 32 → 10
784 → 64 → 10
784 → 128 → 10
784 → 256 → 10


More neurons provide greater model capacity but also increase the number of trainable parameters.

Experiment 3 — Batch Size

Possible batch sizes:

16
32
64
128


A batch size of 32 means that 32 training examples are processed before a parameter update.

Smaller batches generally produce noisier gradient estimates, while larger batches provide more stable estimates but require more memory.

Experiment 4 — Learning Rate

Example learning rates:

0.01
0.001
0.0005
0.0001


The learning rate determines the magnitude of parameter updates.

A learning rate that is too large can cause unstable training, while a learning rate that is too small can make training unnecessarily slow.

Experiment 5 — Number of Epochs

Possible values:

5
10
20
30


One epoch represents one complete pass through the training dataset.

Increasing epochs can improve learning initially, but excessive training may lead to overfitting.

Experiment 6 — Optimizer

Possible optimizers:

SGD
Adam
RMSprop


The performance and convergence behavior of these optimizers can be compared.

Experiment 7 — Dropout

Possible dropout configurations:

No Dropout
Dropout = 0.2
Dropout = 0.5


Dropout randomly deactivates a fraction of activations during training and can help reduce overfitting.

📊 Experimental Results

The actual values should be obtained by running the experiments.

Experiment	Architecture / Configuration	Test Accuracy	Training Time	Observation
Baseline	128 → 64	TBD	TBD	Baseline model
Layer Experiment	128	TBD	TBD	Effect of depth
Neuron Experiment	256 → 128	TBD	TBD	Effect of capacity
Batch Experiment	Batch = 64	TBD	TBD	Effect of batch size
LR Experiment	LR = 0.0001	TBD	TBD	Effect of learning rate
Epoch Experiment	20 epochs	TBD	TBD	Effect of training duration
Dropout Experiment	Dropout = 0.2	TBD	TBD	Effect of regularization

Note: Replace TBD with the results obtained from your actual experiments. Do not use randomly chosen accuracy values.

📈 Evaluation Metrics

The model can be evaluated using:

Accuracy

𝐴
𝑐
𝑐
𝑢
𝑟
𝑎
𝑐
𝑦
=
𝐶
𝑜
𝑟
𝑟
𝑒
𝑐
𝑡
 
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
𝑠
𝑇
𝑜
𝑡
𝑎
𝑙
 
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑖
𝑜
𝑛
𝑠
×
100

Loss

Lower loss generally indicates that the model's predicted probability distribution is closer to the target distribution.

Confusion Matrix

A 10 × 10 confusion matrix can be used to identify which digits the model commonly confuses.

For example:

Actual 5 → Predicted 3


indicates that the model incorrectly classified a 5 as a 3.

🛠️ Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
Scikit-learn
Seaborn
Jupyter Notebook / Google Colab
📁 Project Structure

A recommended repository structure is:

handwritten-digit-recognition/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── mnist_neural_network.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
└── models/
    └── mnist_model.keras

💻 Installation
1. Clone the Repository
git clone <your-repository-url>
cd handwritten-digit-recognition

2. Create a Virtual Environment
python -m venv venv

Windows
venv\Scripts\activate

Linux/macOS
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

📦 Requirements

Example requirements.txt:

tensorflow
numpy
matplotlib
scikit-learn
seaborn
jupyter

▶️ Running the Project

If using the Jupyter Notebook:

jupyter notebook


Then open:

notebooks/mnist_neural_network.ipynb


If using Python scripts:

python src/train.py

🧪 Baseline Implementation

A basic implementation using TensorFlow/Keras:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST
(x_train, y_train), (x_test, y_test) = \
    keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

🧮 Model Parameters

For the baseline architecture:

784 → 128 → 64 → 10


the approximate trainable parameters are:

Layer 1

784
×
128
+
128
=
100
,
480

Layer 2

128
×
64
+
64
=
8
,
256

Output Layer

64
×
10
+
10
=
650

Total

100480
+
8256
+
650
=
109
,
386

Thus, the baseline network learns 109,386 trainable parameters.

📌 Important Concepts Demonstrated

This project demonstrates the following machine-learning and deep-learning concepts:

MNIST Dataset
     ↓
Data Preprocessing
     ↓
Normalization
     ↓
Neural Network
     ↓
Weights & Biases
     ↓
Activation Functions
     ↓
Forward Propagation
     ↓
Loss Function
     ↓
Backpropagation
     ↓
Gradient-based Optimization
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation

⚠️ Overfitting and Underfitting
Underfitting

Occurs when the model is too simple or insufficiently trained.

Training Accuracy → Low
Validation Accuracy → Low


Possible solutions:

Increase model capacity.
Increase training time.
Adjust the learning rate.
Improve the architecture.
Overfitting

Occurs when the model performs very well on training data but generalizes poorly.

Training Accuracy → Very High
Validation Accuracy → Lower


Possible solutions:

Dropout
Early stopping
Regularization
Reducing model complexity
🔮 Future Improvements

The current project uses a fully connected neural network. It can be extended in several ways:

Implement a Convolutional Neural Network (CNN).
Compare Dense and CNN architectures.
Perform automated hyperparameter tuning.
Add data augmentation.
Use early stopping.
Experiment with learning-rate scheduling.
Deploy the trained model as a web application.
Create a real-time handwritten digit recognition interface.
📖 Key Takeaways

The most important concepts learned from this project are:

MNIST images contain 784 pixels.
A neural network learns weights and biases from training data.
Hidden-layer size is a hyperparameter and is not fixed by MNIST.
ReLU provides non-linearity in hidden layers.
Softmax converts output scores into probabilities.
Cross-entropy measures classification error.
Backpropagation calculates gradients.
Optimizers use gradients to update model parameters.
Learning rate controls the size of parameter updates.
Batch size controls how many samples are processed per update.
Epochs determine how many times the model sees the training data.
Hyperparameter experiments help identify an effective model configuration.
Validation performance is important for selecting hyperparameters.
Test data should ideally be reserved for final evaluation.
👨‍💻 Author

Your Name

GitHub: Ayushmaan Tiwari

Project: Handwritten Digit Recognition using Neural Network

📜 License : This project is intended for educational and academic purposes.
