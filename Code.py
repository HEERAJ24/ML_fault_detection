"""""
Support vector machine implementation:
This documentation part showcases the implementation of Support Vector Machines (SVM) for detecting solder joint faults in electronic components. 
SVM is a popular machine learning algorithm that is effective in handling high-dimensional data and has been shown to perform well in various classification tasks. 
In this implementation, the SVM algorithm is trained on a dataset of solder joint sensor data to predict the type of fault present in the joint. 
This code serves as a practical example of how SVMs can be used for solder joint inspection and fault detection.
Below you may find the complete process of data preprocessing, model definition, training, evaluation, and visualization of training loss and accuracy.
"""
# Support Vector Machines (maps data to higher dimension for better segregation)
# CrossEntropyLoss and SGD optimiser
# import the necessary libraries for data preprocessing,
# model definition, and visualization.
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data = pd.read_csv('my_data (2).csv')
#The features (S1, S2, S3, and S4) are extracted and stored in a variable x.
#The labels (Device type) are extracted and stored in a variable y.
x = data[['S1', 'S2', 'S3', 'S4']].values
y = data['Device type'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
"""
Load and Preprocess Data:
data = pd.read_csv('my_data 012.csv') reads the CSV file named 'my_data 012.csv' and stores its contents(in this case solder fault
patterns) in a Pandas DataFrame called data .
The data is loaded from a CSV file, and the relevant features and labels are extracted. The data is then split into training and
testing sets using the train_test_split function. Features are normalized using the StandardScaler , and labels are encoded
using LabelEncoder . The preprocessed data is converted into PyTorch tensors with the appropriate data types.

Split Data:
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
"""
The data is split into training and testing sets using Scikit-learn's train_test_split function. 80% of the data is used for training,
and the remaining 20% is used for testing. The random_state parameter is set to 42 for reproducible results.
Initialize Variables and Normalize Data:
"""
train_losses = []
train_accuracies = []
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
"""
train_losses and train_accuracies are initialized as empty lists. They will be used to store the loss and accuracy values of the
model during training.
The StandardScaler from Scikit-learn is used to normalize the feature data. It is fit to the training data ( x_train ) and then used to
transform both the training and testing data ( x_train and x_test ).

Encode Labels and Convert to Tensors:
"""
#encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
#conver to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
"""
The LabelEncoder from Scikit-learn is used to convert the categorical labels into numerical values. It is fit to the training labels
( y_train ) and then used to transform both the training and testing labels ( y_train and y_test ).
The normalized feature data and encoded labels are converted to PyTorch tensors for further processing. The feature data is
converted to float32 tensors, while the label data is converted to long tensors.
With the data preprocessed and ready, you can now proceed to define a machine-learning model and train it using the
prepared tensors.

Define the SVM Model:
This code demonstrates how to perform device type classification using a support vector machine (SVM) implemented with
PyTorch. It defines an SVM model class, initializes the model, sets up the loss function and optimizer, and then trains the model
using the preprocessed data tensors from the previous code snippet.
"""
class SVM(nn.Module):
   def __init__(self, input_dim, num_classes):
     super(SVM, self).__init__()
     self.linear = nn.Linear(input_dim, num_classes)
   def forward(self, x):
     return self.linear(x)
"""
The SVM class is a subclass of PyTorch's nn.Module . It defines a simple linear model that takes an input dimension ( input_dim )
and the number of classes ( num_classes ). The forward method returns the output of the linear layer.

Initialize the Model:
"""
model = SVM(input_dim=4, num_classes=3)
"""
An instance of the SVM model is created with an input dimension of 4 (the number of features) and 3 classes (the number of
device types).

Set up Loss Function and Optimizer:
"""
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
"""
The loss function is set to CrossEntropyLoss , and the optimizer is set to stochastic gradient descent (SGD) with a learning rate of
0.01 and momentum of 0.9.

Train the Model:
"""
num_epochs = 200
for epoch in range(num_epochs):
 outputs = model(X_train)
 loss = criterion(outputs, y_train)
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 if (epoch+1) % 10 == 0:
  print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
 train_losses.append(loss.item())

 y_pred = model(X_test)
_, predicted = torch.max(y_pred, 1)
accuracy = (predicted == y_test).float().mean().item()
train_accuracies.append(accuracy)
"""
The model is trained for 200 epochs. In each epoch, the model's output is calculated, the loss is computed, and the optimizer
updates the model's parameters. The loss and accuracy values are stored in the train_losses and train_accuracies lists,
respectively.

Evaluate the Model:
"""
with torch.no_grad():
 y_pred = model(X_test)
 _, predicted = torch.max(y_pred, 1)
 accuracy = (predicted == y_test).float().mean().item()
 print('Accuracy: {:.2f}%'.format(accuracy*100))
 train_accuracies.append(accuracy)
 """
Finally, the trained model is evaluated on the test data. The predicted labels are obtained, and the test accuracy is calculated
and printed. The test accuracy is also added to the train_accuracies list.
This code shows how to implement a simple SVM for device type classification using PyTorch. The model's performance can
be further improved by tuning the hyper-parameters or using more advanced techniques.
Plot Training Loss & Plot Training Accuracy
This part of the code visualizes the training loss and training accuracy over time using Matplotlib. It consists of two parts:
plotting the training loss over time and plotting the training accuracy over time. Additionally, there are some comments that
summarize the observed training accuracy for different numbers of epochs and learning rates.
"""
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss over Time')
plt.show()
#plot training accuracy
plt.plot(train_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy over Time')
plt.show()
"""
This part of the code plots the train_losses list, which contains the training loss values for each epoch. The x-axis is labeled as
'Epoch', the y-axis is labeled as 'Training Loss', and the plot is given the title 'Training Loss over Time'. The plt.show()
command displays the plot.
This part of the code plots the train_accuracies list, which contains the training accuracy values for each epoch. The x-axis is
labeled as 'Epoch', the y-axis is labeled as 'Training Accuracy', and the plot is given the title 'Training Accuracy over Time'. The
plt.show() command displays the plot.

Sensor-Based Device Classification with PyTorch:
The code below describes a Python script that trains a neural network to classify device types based on sensor readings using
the PyTorch library. It reads data from a CSV file, preprocesses the data, splits it into training and validation sets, defines a
neural network architecture, and trains the network. Finally, the script plots the training loss and accuracy over time and creates
a scatter plot of predicted vs. true labels. Below you may find step-by-step guidance:
1. Imports necessary libraries and modules.
2. Loads and preprocesses the data.
3. Splits the data into training and validation sets.
4. Defines a neural network architecture.
5. Initializes the model and the optimizer.
6. Trains the network and calculates the training and validation accuracies.
7. Visualizes the results using scatter plots and line plots.

Prerequisites:
To run this script, you will need the following Python libraries installed:
PyTorch
Scikit-learn
Pandas
NumPy
Matplotlib

Importing libraries and modules:
The script imports the necessary libraries and modules for data manipulation, model training, and visualization:
torch: PyTorch library for tensor computation and deep learning.
sklearn: Scikit-learn library for machine learning and data preprocessing.
pandas: Library for data manipulation and analysis.
numpy: Library for numerical computations.
matplotlib: Library for creating visualizations.

Loading and preprocessing the data:
The script reads the data from a CSV file and stores it in a pandas DataFrame. The feature columns (S1, S2, S3, and S4) are
stored in the variable X , and the target column (Device type) is stored in the variable y . The feature data is then standardized
using StandardScaler, and the target data is label-encoded.
"""
#Reads the CSV file containing sensor data and device types, and stores it in a pandas DataFrame.
data = pd.read_csv('my_data 012.csv')
#Extracts the feature columns (S1, S2, S3, and S4) and the target column (Device type) from the DataFrame.
X = data[['S1', 'S2', 'S3', 'S4']].values
y = data['Device type'].values
"""
Dataset:
The dataset used in this script should be in CSV format and consist of the following columns:
S1, S2, S3, S4: Sensor readings (float)
Device type: Labels for device classification (string)
"""
data = pd.read_csv('my_data 012.csv')
X = data[['S1', 'S2', 'S3', 'S4']].values
y = data['Device type'].values
train_losses = []
train_accuracies = []
"""
Preprocessing:
The script preprocesses the data as follows:
1. Normalize the sensor readings (S1, S2, S3, S4) using the StandardScaler from Scikit-learn.
2. Encode the device type labels using the LabelEncoder from Scikit-learn.
3. Split the dataset into training (80%) and validation (20%) sets.
"""
# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
"""
Neural Network Architecture:
The neural network architecture used in this case is a simple feedforward network consisting of two fully connected layers, also
known as linear layers, and an activation function in between. This architecture is designed to solve multi-class classification
problems, as it takes a set of input features and outputs a probability distribution over the different classes.
The neural network architecture is a simple feedforward network with the following layers:
Fully connected layer with input_size (number of features) input units and hidden_size hidden units:
Fully connected layer 1: This layer is also known as a linear layer, and it takes the input features and applies a linear
transformation to them. In this case, the input size is 4, corresponding to the four sensor readings (S1, S2, S3, S4). The
layer has hidden_size (64 in this example) hidden units, which can be adjusted based on the complexity of the data. Each
hidden unit calculates a weighted sum of the input features, followed by an activation function. The output of this layer is a
tensor of shape (batch_size, hidden_size).
LeakyReLU activation function:
The Leaky Rectified Linear Unit (LeakyReLU) is an activation function that introduces nonlinearity into the network. It is
applied element-wise to the output of the first fully connected layer. The LeakyReLU function is defined as: LeakyReLU(x) =
max(0,x) + negative_slope * min(0, x). The negative_slope parameter determines the slope of the function for negative input
values. In this example, the default value of 0.01 is used. This activation function helps the network learn complex patterns
in the data.
Fully connected layer with hidden_size input units and num_classes output units.
1. Fully connected layer 2: This is the output layer of the network. It is another linear layer that takes the output of the
LeakyReLU activation function and applies another linear transformation. The input size of this layer is equal to the
hidden_size of the previous layer, and the output size is num_classes (3 in this example), which corresponds to the number
of possible device types in the classification problem. The output of this layer is a tensor of shape (batch_size, num_classes) .
"""
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)
  def forward(self, x):
    x = nn.LeakyReLU()(self.fc1(x))
    x = self.fc2(x)
    return x
"""
Initialize the network and the optimizer:
1. input_size = 4 : Variable defines the input size of the neural network, which is the number of features in the dataset (S1, S2,
S3, and S4).
2. hidden_size = 64 : This variable defines the number of hidden units in the first fully connected layer of the neural network.
This is a hyperparameter that can be adjusted based on the complexity of the data.
3. num_classes = 3 : This variable defines the number of output classes or categories in the dataset. In this case, the neural
network will be trained to classify devices into three different categories.
4. model = NeuralNet(input_size, hidden_size, num_classes) : This line creates an instance of the NeuralNet class, which is the
neural network architecture defined earlier in the code. The instance is initialized with the specified input size, hidden size,
and a number of classes.
5. criterion = nn.CrossEntropyLoss() : This line defines the loss function for the neural network. In this case, the cross-entropy
loss is used, which is suitable for multi-class classification problems.
6. optimizer = optim.SGD(model.parameters(), lr=0.1) : This line defines the optimizer used for training the neural network. In this
case, Stochastic Gradient Descent (SGD) is used with a learning rate (lr) of 0.1. The optimizer takes the model's
parameters as input, which will be updated during the training process.

Training:
The training part of the code is where the neural network learns to classify the devices based on the input features (sensor
readings). The training process consists of several steps that are repeated for a specified number of epochs. In this code, the
number of epochs is set to 150.
The network is trained using the following parameters:
Loss function: Cross-entropy loss
Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.1
Number of epochs: 150
During each epoch, the script performs the following steps:
1. Forward pass through the network.
2. Calculate the training loss and validation accuracy.
3. Backward pass through the network and update the parameters.
"""
# Train the network
for epoch in range(150):
# Get the outputs and loss for the current iteration
 outputs = model(X_train)
 loss = criterion(outputs, y_train)
 # Calculate the validation accuracy
 val_outputs = model(X_val)
 _, predicted = torch.max(val_outputs.data, 1)
 correct = (predicted == y_val).sum().item()
 val_accuracy = correct / len(y_val)
 # Perform a backward pass and update the parameters
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 # Print the current accuracy and losspre
 _, predicted = torch.max(outputs.data, 1)
 correct = (predicted == y_train).sum().item()
 accuracy = correct / len(y_train)
 train_losses.append(loss.item())
 train_accuracies.append(accuracy)
"""
Here's a detailed explanation of the training process:
1. Loop over epochs: The training process iterates over the specified number of epochs. In each epoch, the network's
parameters are updated using the gradients of the loss function with respect to the parameters.
2. Forward pass: In the forward pass, the input data ( X_train ) is passed through the neural network, and the model's output
is obtained. This output is a probability distribution over the different device types.
3. Calculate the loss: The loss function (cross-entropy loss) is applied to the model's output and the true labels ( y_train ).
This loss measures the difference between the predicted probability distribution and the true labels, and the goal is to
minimize it during the training process.
4. Validation accuracy: To check for overfitting, the validation accuracy is calculated during training. The validation data
( X_val ) is passed through the model, and the predictions are compared with the true validation labels ( y_val ). The number
of correct predictions is divided by the total number of validation samples to obtain the validation accuracy.
5. Backward pass: The gradients of the loss function with respect to the model's parameters are computed using the
loss.backward() function. This step calculates the partial derivatives of the loss function with respect to each parameter,
which is needed for the optimizer to update the parameters.
6. Update the parameters: The optimizer (SGD in this case) updates the model's parameters using the computed gradients.
This is done by calling the optimizer.step() function, which adjusts the parameters according to the optimizer's learning
rate and the gradients.
7. Print the progress: In each epoch, the training accuracy, training loss, and validation accuracy are printed. This helps in
monitoring the training process and observing if the model is overfitting or underfitting the data.
8. Store loss and accuracy values: The training loss and training accuracy for each epoch are stored in train_losses and
train_accuracies lists, respectively. These values can be used later for visualization and analysis.

Results:
The script prints the training accuracy, training loss, and validation accuracy for each epoch. It also plots the training loss and
accuracy over time, providing a visual representation of the model's performance. Finally, it creates a scatter plot of predicted
vs. true labels to visualize the model's predictions.

Potential Overfitting:
This script helps to identify overfitting by plotting the training accuracy and loss over time and printing the validation accuracy
for each epoch. If the training accuracy increases and the training loss decreases while the validation accuracy remains
stagnant or decreases, it could be an indication of overfitting.

Visualizing Classifier Performance with Confusion Matrix Plots:
This script defines a function for plotting a confusion matrix to visualize the performance of a classifier. The confusion matrix
shows the number of true positives, true negatives, false positives, and false negatives for each class, providing insights into
the model's accuracy patterns.

Importing libraries and modules:
In this code we import the necessary libraries and modules for generating confusion matrices and visualizations:
sklearn.metrics : Provides the confusion_matrix function to calculate the confusion matrix.
numpy : A popular library for numerical operations in Python.
matplotlib.pyplot : A plotting library for creating static, animated, and interactive visualizations in Python.
sklearn.utils.multiclass : Provides the unique_labels function, which returns the unique labels from the input data.

Defining the plot_confusion_matrix function:
The function computes the confusion matrix using confusion_matrix(y_train, predicted) , and normalizes it if the normalize
parameter is True . The function then creates a plot of the confusion matrix with appropriate labels, ticks, and color mapping.
The function also formats the text within each cell of the matrix based on the normalize parameter and the threshold value
calculated from the maximum value in the confusion matrix.

Calling the plot_confusion_matrix function:
The script calls the plot_confusion_matrix function twice, first with the normalize parameter set to False (to create a nonnormalized
confusion matrix) and then with it set to True (to create a normalized confusion matrix). The true and predicted
class labels are passed to the function, as well as the class labels from the LabelEncoder object.

Function definition: plot_confusion_matrix:
"""
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
 if not title:
   if normalize:
     title = 'Normalized confusion matrix'
   else:
     title = 'Confusion matrix, without normalization'

   cm = confusion_matrix(y_train, predicted)
   classes = classes[unique_labels(y_train, predicted)]
   if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
   else:
    print('Confusion matrix, without normalization')
   print(cm)
"""
This function is designed to create and display a confusion matrix, which is a visualization that shows the performance of a
classifier. The function takes several parameters:
y_true : The true class labels.
y_pred : The predicted class labels.
classes : An array of class labels.
normalize : A boolean flag indicating whether to normalize the confusion matrix (default is False ).
title : The title of the confusion matrix plot (default is None ).
cmap : The colormap to use for the plot (default is plt.cm.Blues ).
Compute the confusion matrix:
cm = confusion_matrix(y_train, predicted) :calculates the confusion matrix using the true class labels ( y_train ) and the
predicted class labels ( predicted ). The confusion matrix is a square matrix where each row represents the true class and
each column represents the predicted class. The diagonal elements represent the number of correct predictions for each
class.

Normalize the confusion matrix:
"""
if normalize:
 cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
"""
If the normalize parameter is set to True , the function normalizes the confusion matrix by dividing each element in a row by the
sum of that row. This results in a matrix where each row sums up to 1, providing an understanding of the classifier's
performance in terms of proportions or probabilities.

Create the confusion matrix plot and Customize the plot:
"""
#initializes a new matplotlib figure and axis, and then displays the confusion matrix as an image using the specified colormap.
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title=title, ylabel
='True label', xlabel='Predicted label')

#Text formatting and thresholding
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
"""""
Depending on whether the confusion matrix is normalized or not, the text formatting is set to display floating-point numbers with
2 decimal places or integers. The threshold value is calculated as half of the maximum value in the confusion matrix.

Annotate the plot with cell values:
"""""
for i in range(cm.shape[0]):
 for j in range(cm.shape[1]):
  ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
plot_confusion_matrix(predicted, y_train, classes=le.classes_)
"""""
This nested loop iterates over the cells of the confusion matrix and annotates each cell with its value, formatted according
to the fmt variable. The text color is set to white if the cell value is greater than the threshold, and black otherwise. This
enhances the readability of the plot.
Call the plot_confusion_matrix function to generate and display a non-normalized confusion matrix and a normalized
confusion matrix. The true and predicted class labels, as well as the class labels from the LabelEncoder object, are passed
to the function.
plt.show() displays the generated confusion matrix plots (non-normalized and normalized) in separate windows. This
allows for easy comparison and analysis of the classifier's performance.

Random Forest Classifier with Hyperparameter Tuning:
Introduction:
The Random Forest algorithm is a popular machine learning technique that is widely used for classification and regression
tasks. In this report, we present a Python code that utilizes the Scikit-learn library to train and test a Random Forest classifier
on a given dataset. The code implementation involves a step-by-step process, starting with the loading and pre-processing of
the dataset, followed by hyperparameter tuning to optimize the model's performance. We then trained and tested the model on
the dataset, and finally, evaluated its accuracy using the accuracy_score metric. Our report provides a detailed explanation of
each step involved in the implementation of the Random Forest classifier.

Data Processing:
Import the necessary libraries and load the dataset:
"""""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
"""""
Pandas is used to read in the dataset and create a DataFrame. RandomForestClassifier is the algorithm used to train the
model. train_test_split is used to split the data into training and testing sets. GridSearchCV is used to perform a grid search
to find the best hyperparameters for the model. StandardScaler is used to normalize the data. accuracy_score is used to
calculate the accuracy of the model.

Load the dataset:
"""""
data = pd.read_csv('02.05_3sc.csv')
"""""
This code loads the CSV file into a pandas DataFrame called data.

Prepare the data:
"""
X = data[['S1', 'S2', 'S3', 'S4']].values
y = data['cluster'].values
"""
This code separates the features (X) and the labels (y) from the DataFrame.

Split the data into training and testing sets:
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
This code randomly splits the data into a training set and a testing set. The test_size parameter specifies the percentage of the
data that should be used for testing. The random_state parameter sets a seed for the random number generator, which ensures
that the same split is obtained every time the code is run.

Normalize the data:
"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
This code normalizes the data using the StandardScaler function. The fit_transform() method scales the training set, and the
transform() method scales the testing set using the same scaling factors as the training set.

Model training:
1. Define the hyperparameters for the Random Forest model:
"""
param_grid = {
'n_estimators': [50, 100, 200, 300],
'max_depth': [5, 10, 20, 30]
}
"""
This code defines a grid of hyperparameters that will be used to tune the model. The n_estimators parameter specifies the
number of trees in the forest, and the max_depth parameter specifies the maximum depth of each tree.
2. Create a Random Forest classifier object:
"""
#classifier object called rf with a fixed random state of 42
rf = RandomForestClassifier(random_state=42)
"""
3. Perform a grid search to find the best hyperparameters:
"""
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
"""
GridSearchCV performs a grid search over the hyperparameters defined in param_grid using the training set. The cv parameter
specifies the number of cross-validation folds, and n_jobs specifies the number of CPU cores to use (-1 means use all
available cores). The best hyperparameters are stored in the best_params_ attribute of the grid_search object.
4. Print the best hyperparameters and the corresponding accuracy:
"""
print("Best parameters:", grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)
"""
5. Train a Random Forest model with the best hyperparameters:
"""
clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
max_depth=grid_search.best_params_['max_depth'],
random_state=42)
clf.fit(X_train, y_train)
"""
RandomForestClassifier creates a new Random Forest classifier object called clf with the best hyperparameters found by the
grid search. The model is then trained on the training set.
6. Test the model on the testing set:
"""
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
"""
The accuracy of the model is then calculated using the accuracy_score() function, and printed to the console. The accuracy is
expressed as a percentage.

Random Forest Classifier with Oversampling and Undersampling:
In this section we will skip the details that was already explained in Random Forest Classifier with hyper-parameter tuning.
Main difference in this code importing from imblearn.over_sampling import RandomOverSampler and from imblearn.under_sampling import
RandomUnderSampler which import the RandomOverSampler and RandomUnderSampler classes from the imblearn library.
These classes are used to perform oversampling and undersampling, respectively, to address class imbalance in the dataset.
Furthemore, we create instances of the RandomOverSampler and RandomUnderSampler classes with a specified random
state, and then use these classes to resample the data. The oversampling and undersampling steps are performed sequentially
to first increase the number of samples in the minority class and then reduce the number of samples in the majority class. The
final resampled dataset is stored in the variables X_resampled and y_resampled .
"""
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
"""
These lines generate predictions on both the training and testing sets using the predict method of the trained classifier (clf), and
then calculate the accuracy, precision, recall, and F1-score of the predictions for both sets. These metrics are then printed to
the console to evaluate the performance of the model on both sets.
"""
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
y_val_pred = clf.predict(X_test)
val_accuracy = accuracy_score(y_test, y_val_pred)
val_precision = precision_score(y_test, y_val_pred, average='weighted')
val_recall = recall_score(y_test, y_val_pred, average='weighted')
val_f1_score = f1_score(y_test, y_val_pred, average='weighted')
print("Training Set Performance:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-score:", train_f1_score)
print("Validation Set Performance:")
print("Accuracy:", val_accuracy)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1-score:", val_f1_score)

#Below you may see full code of the algorithm to inspect:

#### Random Forest including oversampler and undersampler for balanced classes. Best parameters: {'max_depth': 20, 'n_estimators': 300
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
data = pd.read_csv('02.05_3sc.csv')
X = data[['S1', 'S2', 'S3', 'S4']].values
y = data['cluster'].values
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
param_grid = {
'n_estimators': [300],
'max_depth': [20]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)
clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
max_depth=grid_search.best_params_['max_depth'],
random_state=42)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
y_val_pred = clf.predict(X_test)
val_accuracy = accuracy_score(y_test, y_val_pred)
val_precision = precision_score(y_test, y_val_pred, average='weighted')
val_recall = recall_score(y_test, y_val_pred, average='weighted')
val_f1_score = f1_score(y_test, y_val_pred, average='weighted')
print("Training Set Performance:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-score:", train_f1_score)
print("Validation Set Performance:")
print("Accuracy:", val_accuracy)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1-score:", val_f1_score)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
"""
Label Check and Unique Values Analysis for Random Forest Classifier
Predictions
"""
for true, pred in zip(y_test, y_pred):
 print("True: {:2d} Pred: {:2d}".format(int(true), int(pred)))
print(pd.Series(y_pred).unique())
print(pd.Series(y_test).unique())
"""
This code block prints out the true and predicted labels for each sample in the testing set, and then prints out the unique values
of the predicted and true labels.
The first line uses the zip function to iterate over pairs of true labels (y_test) and predicted labels (y_pred) simultaneously. For
each pair, it prints out the true label and predicted label using the format method of a string. The format method uses integer
placeholders to ensure that the labels are printed as integers, rather than as floating point numbers.
The second line prints out the unique values of the predicted labels using the unique method of a Pandas Series object. This
line is useful for checking if the model is predicting labels other than those seen in the training set. If the model is predicting
new labels, it may be overfitting to the training set or encountering samples that are significantly different from those seen in the
training set.
The third line prints out the unique values of the true labels using the unique method of a Pandas Series object. This line is
useful for checking if the testing set contains labels other than those seen in the training set. If the testing set contains new
labels, it may not be representative of the distribution of labels in the real-world application of the model, and the model's
performance may not generalize well to new data.

Plotting Confusion Matrices for Random Forest Classifier Predictions:
"""
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
y_pred_encoded = le.transform(y_pred)
def plot_confusion_matrix(y_test, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
 if not title:
  if normalize:
   title = 'Normalized confusion matrix'
  else:
   title = 'Confusion matrix, without normalization'
 cm = confusion_matrix(y_test, y_pred)
 unique_labels = np.unique(np.concatenate((y_test, y_pred), axis=0))
 if normalize:
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Normalized confusion matrix")
 else:
  print('Confusion matrix, without normalization')
 print(cm)
 fig, ax = plt.subplots()
 im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
 ax.figure.colorbar(im, ax=ax)
 ax.set(xticks=np.arange(cm.shape[1]),
 yticks=np.arange(cm.shape[0]),
 xticklabels=classes, yticklabels=classes,
 title=title,
 ylabel='True label',
 xlabel='Predicted label')
 plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
 rotation_mode="anchor")
 fmt = '.2f' if normalize else 'd'
 thresh = cm.max() / 2.
 for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
   ax.text(j, i, format(cm[i, j], fmt),
      ha="center", va="center",
      color="white" if cm[i, j] > thresh else "black")
 fig.tight_layout()
 return ax
np.set_printoptions(precision=2)
plot_confusion_matrix(y_test, y_pred, classes=le.classes_,
title='Confusion matrix, without normalization')
plt.show()
plot_confusion_matrix(y_test, y_pred, classes=le.classes_, normalize=True,
title='Normalized confusion matrix')
plt.show()
"""
This section of the code defines a function for plotting a confusion matrix and then uses it to plot both a non-normalized and
normalized confusion matrix for the Random Forest classifier predictions. A confusion matrix is a table used to evaluate the
performance of a classification model by comparing its predicted labels to the true labels.
The first few lines of the code block import the necessary libraries and define a LabelEncoder object to encode the true and
predicted labels.
The plot_confusion_matrix function takes as input the true labels, predicted labels, classes (i.e., unique labels), a boolean flag to
indicate whether or not to normalize the matrix, a title for the plot, and a colormap.
Inside the function, the confusion matrix is calculated using the confusion_matrix function from the sklearn.metrics library. The
function then concatenates the true and predicted labels to get a list of unique labels, which is used to set the x and y tick
labels on the plot.
If the normalize flag is set to True , the confusion matrix is normalized by dividing each row by the sum of the row, and the
function prints out a message indicating that the matrix is normalized.
The imshow method is used to create an image of the confusion matrix, and the colorbar method is used to add a colorbar to
the plot. The set method is used to set the x and y tick labels, title, and axis labels.
The function then iterates over each cell of the matrix and adds text labels to the cells to indicate the value of the cell. The font
color is set to white if the cell value is greater than half the maximum value in the matrix, and black otherwise.
After the plot_confusion_matrix function is defined, it is used to plot two confusion matrices using the plt.show() method. The
first plot shows the non-normalized confusion matrix, and the second plot shows the normalized confusion matrix. These plots
can be used to evaluate the performance of the Random Forest classifier by visualizing the distribution of predicted labels
relative to true labels.

Predicting Solder Fault Clusters with Manual Input Using a Trained Random
Forest Classifier
"""
## Manual input ##
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
data = pd.read_csv('02.05_3sc.csv')
X = data[['S1', 'S2', 'S3', 'S4']].values
y = data['cluster'].values
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
best_params = {'max_depth': 20, 'n_estimators': 300}
clf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
max_depth=best_params['max_depth'],
random_state=42)
clf.fit(X_resampled, y_resampled)
def predict_cluster(s1, s2, s3, s4):
 X_input = scaler.transform([[s1, s2, s3, s4]])
 cluster = clf.predict(X_input)[0]
 return cluster
"""
Random Forest classifier predicts the cluster of solder faults using manual input provided through the predict_cluster function.
The function takes four arguments representing the values of four sensors and returns the predicted cluster of the solder fault.
Solder fault detection PyTorch documentation 17
The first few lines of the code block import the necessary libraries, read the dataset, and split the data into features (X) and
target (y). The data is then resampled using oversampling and undersampling techniques to balance the classes of the target
variable.
The StandardScaler is then used to standardize the resampled features, and the best hyperparameters found from the previous
Random Forest classifier implementation (i.e., best_params ) are used to define a new Random Forest classifier.
The predict_cluster function takes four input arguments representing the values of four sensors. These values are
standardized using the StandardScaler object and transformed into a two-dimensional array to match the shape of the training
data. The predict method of the clf object is then used to predict the cluster of the solder fault.
The function returns the predicted cluster as a single value, which can be used to make decisions about the quality of the
solder joint.

Scaling and Standardizing Input Data for Accurate Device Type Predictions:
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
new_data = pd.read_csv('my_data 012.csv')
new_data1 = pd.read_csv('my_data (2).csv')
X_new = new_data[['S1', 'S2', 'S3', 'S4']].values
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)
y_pred = clf.predict(X_new)
results_df = pd.DataFrame(new_data, columns=['S1', 'S2', 'S3', 'S4', 'Device type'])
results_df1 = pd.DataFrame(new_data1, columns=['Device type1'])
results_df['prediction'] = y_pred
results_df['Device type'] = results_df['Device type'] + 1
true_values = new_data['Device type'].values
correct_predictions = (y_pred == true_values).sum()
total_predictions = len(true_values)
accuracy = correct_predictions / total_predictions * 100
print("Accuracy: {:.2f}%".format(accuracy))
print(results_df)
print(results_df1)
"""
We read the two new datasets and apply a trained Random Forest classifier to predict the device type based on the sensor
data. The code starts with importing the necessary libraries, including pandas, StandardScaler, and joblib. It then sets some
options for the pandas display to allow all rows and columns to be displayed. Next, the code reads in the two new datasets
using the pd.read_csv method and extracts the sensor data from the new_data DataFrame to create a new feature matrix, X_new .
The StandardScaler is then used to standardize the feature matrix, and the predict method of the trained classifier clf is used
to predict the device type based on the sensor data.
A DataFrame called results_df is then created to store the predicted device type, along with the sensor data and the original
device type from new_data . Another DataFrame called results_df1 is created to store the device type from new_data1 .
The predicted device type is added as a new column to results_df , and the true device type is incremented by 1 to match the
indexing of the predicted device type. The accuracy of the predictions is then calculated by comparing the predicted device
type to the true device type, and calculating the percentage of correct predictions.
Finally, the results_df and results_df1 DataFrames are printed to display the sensor data, predicted device type, and actual
device type.
"""
