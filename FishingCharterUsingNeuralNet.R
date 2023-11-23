# The following code imports a dataset fishing charter and generates a neural
# network to predict if boat used for fishing is a charter boat or not based on
# the Catch Rate and Annual Income

# Installing the tidyverse and neuralnet packages
# install.packages("tidyverse")
# install.packages("neuralnet")

# Loading the tidyverse and neuralnet libraries
library(tidyverse)
library(neuralnet)

# Setting the working directory to your Lab12 folder
setwd("C:/Users/ual-laptop/Desktop/MIS545/Lab12")

# Reading FishingCharter.csv into a tibble called fishingCharter
fishingCharter <- read_csv("FishingCharter.csv",
                         col_types = "lnn",
                         col_names = TRUE)

# Displaying fishingCharter in the console
print(fishingCharter)

# Displaying the structure of fishingCharter in the console
print(str(fishingCharter))

# Displaying the summary of fishingCharter in the console
print(summary(fishingCharter))

# Scaling the AnnualIncome and CatchRate variables
fishingCharter <- fishingCharter %>%
  mutate(AnnualIncomeScaled = (AnnualIncome - min(AnnualIncome))/
           (max(AnnualIncome)- min(AnnualIncome)))
fishingCharter <- fishingCharter %>%
  mutate(CatchRateScaled = (CatchRate - min(CatchRate))/
           (max(CatchRate)- min(CatchRate)))

# Randomly splitting the dataset into fishingCharterTraining (75% of records) 
# and fishingCharterTesting (25% of records) using 591 as the random seed
set.seed(591)
sampleSet <- sample(nrow(fishingCharter),
                    round(nrow(fishingCharter) * 0.75),
                    replace = FALSE)
fishingCharterTraining <- fishingCharter[sampleSet, ]
fishingCharterTesting <- fishingCharter[-sampleSet, ]

# Generating the neural network model to predict CharteredBoat 
# (dependent variable) using AnnualIncomeScaled and 
# CatchRateScaled (independent variables).
fishingCharterNeuralNet <- neuralnet(
  formula = CharteredBoat ~ AnnualIncomeScaled + CatchRateScaled,
  data = fishingCharterTraining,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE
)

# Displaying the neural network numeric results
print(fishingCharterNeuralNet$result.matrix)

# Visualizing the neural network
plot(fishingCharterNeuralNet)

# Using fishingCharterNeuralNet to generate probabilities on the 
# fishingCharterTesting data set and store this in fishingCharterProbability
fishingCharterProbability <- compute(fishingCharterNeuralNet,
                                     fishingCharterTesting)

# Displaying the probabilities from the testing dataset on the console
print(fishingCharterProbability$net.result)

# Converting probability predictions into 0/1 predictions and store this into 
# fishingCharterPrediction
fishingCharterPrediction <-
  ifelse(fishingCharterProbability$net.result > 0.5, 1, 0 )

# Displaying the 0/1 predictions on the console
print(fishingCharterPrediction)

# Evaluating the model by forming a confusion matrix
fishingCharterConfusionMatrix <- table(fishingCharterTesting$CharteredBoat,
                                       fishingCharterPrediction)

# Displaying the confusion matrix on the console
print(fishingCharterConfusionMatrix)

# Calculating the model predictive accuracy
predictiveAccuracy <- sum(diag(fishingCharterConfusionMatrix)) /
  nrow(fishingCharterTesting)

# Displaying the predictive accuracy on the console
print(predictiveAccuracy)
