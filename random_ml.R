#####Machine learning with R

# Load necessary libraries
library(caret)
library(randomForest)
library(rpart)
library(xgboost)  # For XGBoost
library(MASS)  # For Logistic Regression
library(ROCR)  # For ROC and AUC
library(dplyr)
library(ggplot2)
library(corrplot)

set.seed(123)

# Simulate data: 120 observations, 14 predictor variables (mix of categorical, binary, and continuous)
n <- 120

#create id
id <- 1:n

# 4 categorical variables (with 4 groups each)
cat_var1 <- sample(c("A", "B", "C", "D"), n, replace = TRUE)
cat_var2 <- sample(c("W", "X", "Y", "Z"), n, replace = TRUE)

# 2 binary variables
bin_var1 <- sample(c(0, 1), n, replace = TRUE)
bin_var2 <- sample(c(0, 1), n, replace = TRUE)

# 8 continuous variables
cont_var1 <- rnorm(n, mean = 50, sd = 10)
cont_var2 <- rnorm(n, mean = 100, sd = 20)
cont_var3 <- rnorm(n, mean = 75, sd = 15)
cont_var4 <- rnorm(n, mean = 60, sd = 12)
cont_var5 <- rnorm(n, mean = 40, sd = 10)
cont_var6 <- rnorm(n, mean = 42, sd = 8)
cont_var7 <- rnorm(n, mean = 30, sd = 7)
cont_var8 <- rnorm(n, mean = 32, sd = 9)

# Binary outcome variable
outcome <- sample(c(0, 1), n, replace = TRUE)

# Create a data frame
data_raw <- data.frame(id, cat_var1, cat_var2, bin_var1, bin_var2, 
                   cont_var1, cont_var2, cont_var3, cont_var4,
                   cont_var5, cont_var6, cont_var7, cont_var8,
                   outcome)

# View first few rows
head(data_raw)


#remove unneeded object
rm(list = ls(pattern = "^cont"))
rm(list = ls(pattern = "^bin"))
rm(list = ls(pattern = "^cat"))
rm(id)
########
#stage 2


# Explore to find the significant predictors

#plot correlation matrix
# Example of how to exclude specific categorical integer columns
continuous_vars <- data_raw %>%
  select_if(function(x) is.numeric(x) & length(unique(x)) > 4) %>%   # Assume more than 4 unique values indicates continuous
  select(-id)



# check correlation matrix and plot to identify possible autocorrelation
cor_matrix <- cor(continuous_vars)
corrplot(cor_matrix, method = "color", type = "full", 
         col = colorRampPalette(c("red", "white", "blue"))(200), 
         addCoef.col = "black",  
         tl.col = "black", tl.srt = 45)


# Logistic regression model
log_model <- glm(outcome ~ ., data = data_raw %>% select(-id), family = binomial)

# Summarize the model
summary(log_model)

#significant variable are cat_var2, cont_var_8,
#near signif are catvar1, cont_var4, cont_var_6
#remove cont_var3, 5, 7


#Select significant variables at this stage
# Subset data to only relevant variables
selected_vars <- c("id", "bin_var1", "cat_var1", "cont_var1", "cont_var2", "cont_var4", "cont_var6", "cont_var8", "outcome")
data <- data_raw[, selected_vars]


# Split the data into train (80%) and test (20%) sets
set.seed(123)
trainIndex <- createDataPartition(data$outcome, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]


# #Standardisation and one-hot encoding

# # Manually specify the 'cont' variables
cont_vars <- c("cont_var1", "cont_var2", "cont_var4", "cont_var6", "cont_var8")

# Apply preprocessing (center and scale) only to the specified 'cont' variables
preProcValues <- preProcess(trainData[, cont_vars], method = c("center", "scale"))

# Apply the transformations to both trainData and testData
trainData[, cont_vars] <- predict(preProcValues, trainData[, cont_vars])
testData[, cont_vars] <- predict(preProcValues, testData[, cont_vars])


# One-hot encode categorical variables
DummyModel <- dummyVars(" ~ .", data = trainData, fullRank = TRUE)
trainData <- data.frame(predict(DummyModel, newdata = trainData))
testData <- data.frame(predict(DummyModel, newdata = testData))

# View the processed data
head(trainData)



# Logistic regression model
log_model <- glm(outcome ~ ., data = trainData %>% select(-id), family = binomial)

# Summarize the model
summary(log_model)


# Metrics train data
log_train_pred <- predict(log_model, newdata = trainData, type = "response")
log_train_class <- ifelse(log_train_pred > 0.5, 1, 0)
confusionMatrix(as.factor(log_train_class), as.factor(trainData$outcome))



# Predictions on test data
log_pred <- predict(log_model, newdata = testData, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

#create new df with new columns -- log_pred and log_pred_class
testDataNewDf <- data.frame(testData,log_pred, log_pred_class )


# Evaluate performance
confusionMatrix(as.factor(log_pred_class), as.factor(testData$outcome))



log_train_accuracy <- mean(log_train_class == trainData$outcome)
log_test_accuracy <- mean(log_pred_class == testData$outcome)

cat("Logistic Regression - Train Accuracy:", log_train_accuracy, "\n")
cat("Logistic Regression - Test Accuracy:", log_test_accuracy, "\n")



# Random Forest model
rf_model <- randomForest(as.factor(outcome) ~ ., data = trainData, importance = TRUE)

# Variable importance
importance(rf_model)
varImpPlot(rf_model)

# Predictions on test data
rf_pred <- predict(rf_model, newdata = testData, type = "class")

# Evaluate performance
confusionMatrix(rf_pred, as.factor(testData$outcome))

# Metrics
rf_train_pred <- predict(rf_model, newdata = trainData, type = "class")
rf_train_accuracy <- mean(rf_train_pred == trainData$outcome)
rf_test_accuracy <- mean(rf_pred == testData$outcome)

cat("Random Forest - Train Accuracy:", rf_train_accuracy, "\n")
cat("Random Forest - Test Accuracy:", rf_test_accuracy, "\n")



# Decision tree model
dt_model <- rpart(as.factor(outcome) ~ ., data = trainData, method = "class")

# Plot the tree
plot(dt_model)
text(dt_model)

# Predictions on test data
dt_pred <- predict(dt_model, newdata = testData, type = "class")

# Evaluate performance
confusionMatrix(dt_pred, as.factor(testData$outcome))

# Metrics
dt_train_pred <- predict(dt_model, newdata = trainData, type = "class")
dt_train_accuracy <- mean(dt_train_pred == trainData$outcome)
dt_test_accuracy <- mean(dt_pred == testData$outcome)

cat("Decision Tree - Train Accuracy:", dt_train_accuracy, "\n")
cat("Decision Tree - Test Accuracy:", dt_test_accuracy, "\n")



# XGBoost
train_matrix <- model.matrix(outcome ~ . - 1, data = trainData)
train_label <- as.numeric(as.character(trainData$outcome))
test_matrix <- model.matrix(outcome ~ . - 1, data = testData)
test_label <- as.numeric(as.character(testData$outcome))


xgb_model <- xgboost(data = train_matrix, label = train_label, nrounds = 100, objective = "binary:logistic", verbose = 0,
                     eta = 0.1,   # Learning rate
                     max_depth = 3)



xgb_pred <- predict(xgb_model, test_matrix)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)


# Make predictions on test data
xgb_pred <- predict(xgb_model, newdata = test_matrix)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)  # Convert probabilities to binary class labels

# Evaluate performance with confusion matrix
confusionMatrix(as.factor(xgb_pred_class), as.factor(test_label))

# Calculate accuracy metrics
xgb_train_pred <- predict(xgb_model, newdata = train_matrix)
xgb_train_pred_class <- ifelse(xgb_train_pred > 0.5, 1, 0)
xgb_train_accuracy <- mean(xgb_train_pred_class == train_label)
xgb_test_accuracy <- mean(xgb_pred_class == test_label)

# Display results
cat("XGBoost Model - Train Accuracy:", xgb_train_accuracy, "\n")
cat("XGBoost Model - Test Accuracy:", xgb_test_accuracy, "\n")



# Create a new dataset (using some values from the original test set)
new_data <- testData[1:10, ]
new_data$outcome <- NULL

# Logistic regression prediction
log_new_pred <- predict(log_model, newdata = new_data, type = "response")


# Random Forest prediction
rf_new_pred <- predict(rf_model, newdata = new_data, type = "prob")

# Decision Tree prediction
dt_new_pred <- predict(dt_model, newdata = new_data, type = "prob")



#generate new data frame
log_new_pred <- data.frame(log_new_pred)
log_new_pred$id <- as.numeric(rownames(log_new_pred))
log_new_pred$logclass <- ifelse(log_new_pred$log_new_pred > 0.5, 1, 0)


rf_new_pred <- as.data.frame(rf_new_pred)
rf_new_pred$id <- as.numeric(rownames(rf_new_pred))
rf_new_pred <- rf_new_pred %>% select(-`0`) %>% rename(rf_pred =`1`)


dt_new_pred <- as.data.frame(dt_new_pred)
dt_new_pred$id <- as.numeric(rownames(dt_new_pred))
dt_new_pred <- dt_new_pred %>% select(-`0`) %>% rename(dt_pred =`1`)



new_data_all <- new_data %>%
  left_join(rf_new_pred, by = "id") %>%
  left_join(dt_new_pred, by = "id") %>%
  left_join(log_new_pred, by = "id")




#######
#Test prediction with new_test_data similar to raw data
set.seed(123)
new_test_data <- data_raw %>%
  sample_n(30)

#select vars used for prediction
new_test_data_preproc <- new_test_data[, selected_vars]


# Apply the transformations to new_testData
new_test_data_preproc[, cont_vars] <- predict(preProcValues, new_test_data_preproc[, cont_vars])


# One-hot encode categorical variables
new_test_data_preproc <- data.frame(predict(DummyModel, newdata = new_test_data_preproc))


# Predictions on test data
log_pred <- predict(log_model, newdata = new_test_data_preproc, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

#create new df with new columns -- log_pred and log_pred_class
new_test_data_pred_df <- data.frame(new_test_data_preproc,log_pred, log_pred_class )

# Evaluate performance   #This step wont likely be needed since no outcome in expected data
#confusionMatrix(as.factor(log_pred_class), as.factor(new_test_data_preproc$outcome))  



#try the prediction on non processed df
#result is a disaster #lesson - always process your unseen data.


            
