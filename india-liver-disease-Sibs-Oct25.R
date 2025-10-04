# ============================================
# Liver Disease Prediction Project
# ============================================

# 1. Install and load necessary packages
# (This will install any missing packages and then load them)
packages <- c("tidyverse", "caret", "randomForest", "e1071", "ROSE", "pROC")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 2. Define a function to calculate the mode (most common value)
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# 3. Download the dataset if it doesn't already exist in working folder
url <- "https://raw.githubusercontent.com/bizysiby/DS_India_liver_patients/main/indian_liver_patient.csv"
destfile <- "indian_liver_patient.csv"
if (!file.exists(destfile)) {
  download.file(url, destfile, method = "libcurl")
}

# 4. Read the data
data <- read.csv(destfile)

# 5. Data cleaning: Fill in missing values
for (col in names(data)) {
  if (any(is.na(data[[col]]))) {
    if (is.numeric(data[[col]])) {
      # For numbers, use the median
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    } else {
      # For text/factor, use the mode
      data[[col]][is.na(data[[col]])] <- Mode(data[[col]])
    }
  }
}

# 6. Convert Gender to a factor (category), and Dataset to labels
data$Gender <- as.factor(data$Gender)
data$Dataset <- factor(ifelse(data$Dataset == 1, "LiverPatient", "Healthy"))

# 7. Print some basic info about the data
print("Summary of the data:")
print(summary(data))
print("Number of patients in each group:")
print(table(data$Dataset))

# 8. (Optional) Plot some basic charts if to see the data visually
library(ggplot2)
ggplot(data, aes(x = Age)) + geom_histogram(binwidth = 5, fill = "steelblue") + theme_minimal()
ggplot(data, aes(x = Dataset, y = Total_Bilirubin, fill = Dataset)) +
geom_boxplot() + theme_minimal()
ggplot(data, aes(x = Dataset, fill = Dataset)) +
geom_bar() + theme_minimal()

# 9. Balance the classes using ROSE, so both groups are similar in size
data_balanced <- ROSE(Dataset ~ ., data = data, seed = 1)$data
print("Number of patients in each group after balancing:")
print(table(data_balanced$Dataset))

# 10. Split the data into training and testing sets (80% train, 20% test)
set.seed(123)
trainIndex <- createDataPartition(data_balanced$Dataset, p = 0.8, list = FALSE)
train <- data_balanced[trainIndex, ]
test <- data_balanced[-trainIndex, ]

# 11. Prepare scaled data for SVM (Support Vector Machine)
# This is needed because SVM works better with scaled data
scaling_vars <- setdiff(names(train), c("Gender", "Dataset"))
preProc <- preProcess(train[, scaling_vars], method = c("center", "scale"))
train_scaled <- train
test_scaled <- test
train_scaled[, scaling_vars] <- predict(preProc, train[, scaling_vars])
test_scaled[, scaling_vars] <- predict(preProc, test[, scaling_vars])

# 12. Train a Logistic Regression model
set.seed(123)
log_model <- train(Dataset ~ ., data = train, method = "glm", family = "binomial",
                   trControl = trainControl(method = "cv", number = 5, classProbs = TRUE))
log_pred <- predict(log_model, test)
log_prob <- predict(log_model, test, type = "prob")[,2]
log_cm <- confusionMatrix(log_pred, test$Dataset)
log_roc <- roc(response = test$Dataset, predictor = log_prob, levels = rev(levels(test$Dataset)))
print("Logistic Regression Results:")
print(log_cm)
print(paste("Logistic Regression AUC:", round(auc(log_roc), 3)))

# 13. Train a Random Forest model
set.seed(123)
rf_model <- train(Dataset ~ ., data = train, method = "rf",
                  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE))
rf_pred <- predict(rf_model, test)
rf_prob <- predict(rf_model, test, type = "prob")[,2]
rf_cm <- confusionMatrix(rf_pred, test$Dataset)
rf_roc <- roc(response = test$Dataset, predictor = rf_prob, levels = rev(levels(test$Dataset)))
print("Random Forest Results:")
print(rf_cm)
print(paste("Random Forest AUC:", round(auc(rf_roc), 3)))

# See which features are important:
varImpPlot(rf_model$finalModel, main = "Random Forest Feature Importance")

# 14. Train a Support Vector Machine (SVM) model
svm_trctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = TRUE)
set.seed(123)
svm_model <- train(Dataset ~ ., data = train_scaled, method = "svmRadial",
                   trControl = svm_trctrl, tuneLength = 3)
svm_pred <- predict(svm_model, test_scaled)
# Sometimes SVM has trouble with probabilities, so using tryCatch
svm_prob <- tryCatch({
  predict(svm_model, test_scaled, type = "prob")[,2]
}, error = function(e) {
  as.numeric(svm_pred == "LiverPatient")
})
svm_cm <- confusionMatrix(svm_pred, test_scaled$Dataset)
svm_roc <- roc(response = test_scaled$Dataset, predictor = svm_prob, levels = rev(levels(test_scaled$Dataset)))
print("SVM Results:")
print(svm_cm)
print(paste("SVM AUC:", round(auc(svm_roc), 3)))

# 15. Although optional, plot ROC curves for all models on the same plot
plot(log_roc, col = "blue", main = "ROC Curves")
plot(rf_roc, col = "red", add = TRUE)
plot(svm_roc, col = "green", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "SVM"),
        col = c("blue", "red", "green"), lwd = 2)


# --- Collect results for summary table ---

# Logistic Regression metrics
log_acc <- log_cm$overall["Accuracy"]
log_sens <- log_cm$byClass["Sensitivity"]
log_spec <- log_cm$byClass["Specificity"]
log_auc <- as.numeric(auc(log_roc))

# Random Forest metrics
rf_acc <- rf_cm$overall["Accuracy"]
rf_sens <- rf_cm$byClass["Sensitivity"]
rf_spec <- rf_cm$byClass["Specificity"]
rf_auc <- as.numeric(auc(rf_roc))

# SVM metrics
svm_acc <- svm_cm$overall["Accuracy"]
svm_sens <- svm_cm$byClass["Sensitivity"]
svm_spec <- svm_cm$byClass["Specificity"]
svm_auc <- as.numeric(auc(svm_roc))

# Combine into a data frame
results_table <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM"),
  Accuracy = c(log_acc, rf_acc, svm_acc),
  Sensitivity = c(log_sens, rf_sens, svm_sens),
  Specificity = c(log_spec, rf_spec, svm_spec),
  AUC = c(log_auc, rf_auc, svm_auc)
)

# Only round the numeric columns (Accuracy, Sensitivity, Specificity, AUC)
results_table_rounded <- results_table
results_table_rounded[, -1] <- round(results_table_rounded[, -1], 3)

# Print the results table
print("Summary of Model Performance:")
print(results_table_rounded)

# Results table - another type
if (require(knitr)) {
  knitr::kable(results_table_rounded, caption = "Summary of how three Models performed")
}


# 17. End of script
print("Three models trained and evaluated for liver disease prediction using Indian Liver Patient Records collected from North East of Andhra Pradesh, India. Available from https://www.kaggle.com/datasets/uciml/indian-liver-patient-records")