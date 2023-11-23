# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Setting the working directory to your Lab12 folder
# Replace the path with the correct path to your Lab12 folder
import os
os.chdir("C:/Users/ual-laptop/Desktop/MIS545/Lab12")

# Reading FishingCharter.csv into a DataFrame called fishing_charter
fishing_charter = pd.read_csv("FishingCharter.csv")

# Displaying fishing_charter in the console
print(fishing_charter)

# Displaying the structure of fishing_charter in the console
print(fishing_charter.info())

# Displaying the summary of fishing_charter in the console
print(fishing_charter.describe())

# Scaling the AnnualIncome and CatchRate variables
scaler = MinMaxScaler()
fishing_charter['AnnualIncomeScaled'] = scaler.fit_transform(fishing_charter[['AnnualIncome']])
fishing_charter['CatchRateScaled'] = scaler.fit_transform(fishing_charter[['CatchRate']])

# Randomly splitting the dataset into training (75% of records) 
# and testing (25% of records) using 591 as the random seed
fishing_charter_training, fishing_charter_testing = train_test_split(
    fishing_charter, test_size=0.25, random_state=591)

# Generating the neural network model to predict CharteredBoat 
# (dependent variable) using AnnualIncomeScaled and 
# CatchRateScaled (independent variables).
neural_net_model = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic', max_iter=10000)
neural_net_model.fit(fishing_charter_training[['AnnualIncomeScaled', 'CatchRateScaled']],
                     fishing_charter_training['CharteredBoat'])

# Visualizing the neural network
plt.figure(figsize=(10, 6))
plt.title('Neural Network Model')
plt.imshow(neural_net_model.coefs_[0], interpolation='none', cmap='viridis')
plt.colorbar(orientation='vertical', shrink=0.8)
plt.xticks(range(len(fishing_charter.columns)-1),
           fishing_charter.columns[:-1], rotation=90)
plt.yticks(range(3), range(1, 4))
plt.show()

# Using neural_net_model to generate predictions on the testing dataset
fishing_charter_prediction = neural_net_model.predict(
    fishing_charter_testing[['AnnualIncomeScaled', 'CatchRateScaled']])

# Displaying the predictions on the console
print(fishing_charter_prediction)

# Evaluating the model by forming a confusion matrix
fishing_charter_confusion_matrix = confusion_matrix(
    fishing_charter_testing['CharteredBoat'], fishing_charter_prediction)

# Displaying the confusion matrix on the console
print(fishing_charter_confusion_matrix)

# Calculating the model predictive accuracy
predictive_accuracy = accuracy_score(
    fishing_charter_testing['CharteredBoat'], fishing_charter_prediction)

# Displaying the predictive accuracy on the console
print(predictive_accuracy)
