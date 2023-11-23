# Fishing Charter Neural Network Classifier

This repository houses a Python script exemplifying the construction and training of a simple neural network model for classifying whether an individual will charter a fishing boat based on their annual income and catch rate.

## Dependencies
Before running the script, ensure you have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- keras

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn keras
```

## Usage
1. Clone this repository:

```bash
git clone https://github.com/your-username/fishing-charter-predictor-neural-net.git
```

2. Navigate to the repository directory:

```bash
cd fishing-charter-classifier
```

3. Run the Python script:

```bash
python FishingCharterNeuralNetPython.py
```

## Overview
The script performs the following steps:


1. Reads a CSV file named "FishingCharter.csv" into a pandas DataFrame.
2. Scales the 'AnnualIncome' and 'CatchRate' variables using Min-Max scaling.
3. Splits the dataset into a training set (75%) and a testing set (25%).
4. Defines a simple neural network model using Keras with an input layer of 2 neurons, one hidden layer with 3 neurons, and an output layer with 1 neuron. The activation function used is the sigmoid function.
5. Compiles the model using binary cross-entropy loss and the Adam optimizer.
6. Trains the model on the training data for 100 epochs.
7. Uses the trained model to make predictions on the testing data.
8. Displays the probability predictions and 0/1 predictions.
9. Evaluates the model using a confusion matrix and calculates the predictive accuracy.

## Dataset
The dataset used in this script is assumed to be in a CSV file named "FishingCharter.csv" with the following columns:

1. AnnualIncome: The annual income of the individuals.
2. CatchRate: The catch rate of the individuals.
3. CharteredBoat: Whether the individual chartered a fishing boat (1 for chartered, 0 for not chartered).

You can replace this dataset with your own data by providing a CSV file with the same structure.

## Model
The neural network model used is a basic feedforward neural network with a sigmoid activation function. You can modify the architecture of the model by changing the number of hidden layers and neurons as needed.
