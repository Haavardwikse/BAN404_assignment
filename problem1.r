## Nessesary libraries

# Loading necessary libraries
library(Ecdat)       # For the Wages dataset
library(ggplot2)     # For data visualization
library(dplyr)       # For data manipulation
library(caret)       # For data splitting, training, and cross-validation
library(glmnet)      # For Ridge regression and LASSO
library(mgcv)        # For generalized additive models (GAM)
library(rpart)       # For regression trees
library(rpart.plot)  # For plotting regression trees
library(MASS)        # For linear discriminant analysis (LDA)
library(randomForest) # For random forest models, optional for advanced analysis
library(patchwork)
library(caret) # For logistic regression


# Problem 1

## a)


# Load the Wages dataset
data(Wages)

# Explore the structure of the dataset to identify potential predictors
str(Wages)

# List of predictors you want to plot against lwage
selected_predictors1 <- c( "smsa", "married", "sex", "union", "black")
selected_predictors2 <- c("exp", "wks", "bluecol", "ind", "south")
# "exp", "wks", "bluecol", "ind", "south", "smsa", "married", "sex", "union", "black"

plots_list1 <- list()
plots_list2 <- list()

for(predictor in selected_predictors1) {
  plots_list1[[predictor]] <- ggplot(Wages, aes(x = !!sym(predictor), y = lwage)) +
    geom_point(alpha=0.2) +
    geom_smooth(method="lm", se=FALSE, color="blue") +
    labs(title=paste("lwage vs.", predictor), x=predictor, y="Log Wage")
}

for(predictor in selected_predictors2) {
  plots_list2[[predictor]] <- ggplot(Wages, aes(x = !!sym(predictor), y = lwage)) +
    geom_point(alpha=0.2) +
    geom_smooth(method="lm", se=FALSE, color="blue") +
    labs(title=paste("lwage vs.", predictor), x=predictor, y="Log Wage")
}

# Combine and display the plots
combined_plot1 <- wrap_plots(plots_list1, ncol = 3)
combined_plot2 <- wrap_plots(plots_list2, ncol = 3)

# Display the combined plots
combined_plot1
combined_plot2

# Choose predictors "exp", "wks", "bluecol", "ind", "south", "smsa", "married", "sex", "union", "black"
selected_predictors <- c( "exp", "bluecol", "smsa", "married", "sex", "black")

# Dynamically build the formula with polynomial terms for 'exp' and 'ed'
formula_str <- paste("lwage ~", paste(selected_predictors, collapse=" + "))

# Convert string to formula
formula <- as.formula(formula_str)

# Fit the linear regression model with polynomial terms
model <- lm(formula, data=Wages)
summary(model)


# Interpret the model output
# Look at the coefficients to understand the effect of each predictor on lwage
# Pay special attention to the predictors with non-linear terms (e.g., I(predictor1^2))
# Assess the model's overall fit (e.g., R-squared) and individual predictors' significance (p-values)



## d)

```{r, eval=T, error=TRUE, message=TRUE}

# Correcting the function f
f <- function(x0, x, y, K = 20) {
  n <- nrow(x)
  p <- ncol(x)
  d <- matrix(0, n, 1)
  d <- sqrt(apply((x - matrix(x0, n, p, byrow = TRUE))^2, 1, sum))
  o <- order(d)[1:K]
  ypred <- mean(y[o])
  return(ypred)
}

# Encoding 'sex' as a numeric variable where male = 1 and female = 0
Wages$sex_numeric <- ifelse(Wages$sex == "male", 1, 0)

# Splitting the dataset into training and test sets
set.seed(123) # For reproducibility
indices <- sample(1:nrow(Wages), size = 0.8 * nrow(Wages), replace = FALSE) # 80% for training
train_data <- Wages[indices, ]
test_data <- Wages[-indices, ]

# Extracting the chosen predictors ('sex_numeric' and 'ed') and the target variable
x_train <- as.matrix(train_data[, c("sex_numeric", "ed")])
y_train <- train_data$lwage

x_test <- as.matrix(test_data[, c("sex_numeric", "ed")])
y_test <- test_data$lwage

# Initialize vectors to store predictions
predictions_k10 <- numeric(length(y_test))
predictions_k20 <- numeric(length(y_test))

# Loop over the test set for K = 10 and K = 20
for (i in 1:nrow(x_test)) {
  predictions_k10[i] <- f(x0 = x_test[i, ], x = x_train, y = y_train, K = 10)
  predictions_k20[i] <- f(x0 = x_test[i, ], x = x_train, y = y_train, K = 20)
}

# Evaluate the predictions using Mean Squared Error (MSE)
mse_k10 <- mean((predictions_k10 - y_test)^2)
mse_k20 <- mean((predictions_k20 - y_test)^2)

print(paste("MSE for K=10:", mse_k10))
print(paste("MSE for K=20:", mse_k20))


## e)


# Define a function for k-fold cross-validation on the k-NN model
cross_validate_knn <- function(K, folds, x, y) {
  fold_size <- nrow(x) / folds
  mse_list <- numeric(folds)
  
  for (i in 1:folds) {
    # Define indices for the validation set
    val_indices <- (((i - 1) * fold_size + 1):(i * fold_size))
    
    # Split the data into training and validation sets
    x_train_cv <- x[-val_indices, ]
    y_train_cv <- y[-val_indices]
    x_val_cv <- x[val_indices, ]
    y_val_cv <- y[val_indices]
    
    # Predict using the current value of K
    predictions <- sapply(1:nrow(x_val_cv), function(j) f(x0 = x_val_cv[j, ], x = x_train_cv, y = y_train_cv, K = K))
    
    # Calculate MSE for the current fold and store it
    mse_list[i] <- mean((predictions - y_val_cv)^2)
  }
  
  # Return the average MSE across all folds
  return(mean(mse_list))
}

# Range of K values to test
K_values <- seq(15, 30, by = 2)  # Testing K values from 1 to 30, skipping by 2
folds <- 3  # Number of folds in k-fold cross-validation

# Prepare the matrix of predictors and the target variable from the training data
x_train_cv <- as.matrix(train_data[, c("sex_numeric", "ed")])
y_train_cv <- train_data$lwage

# Calculate average MSE for each K
mse_results <- sapply(K_values, function(K) cross_validate_knn(K, folds, x_train_cv, y_train_cv))

# Identify the optimal K with the lowest MSE
optimal_K <- K_values[which.min(mse_results)]
optimal_MSE <- min(mse_results)

print(paste("Optimal K:", optimal_K))
print(paste("Optimal MSE:", optimal_MSE))

# You can then use this optimal K to make predictions and evaluate them as before


## f)


# Assuming 'train_data' and 'test_data' are already defined
x_train <- as.matrix(train_data[, c("sex_numeric", "ed")])
y_train <- train_data$lwage

x_test <- as.matrix(test_data[, c("sex_numeric", "ed")])
y_test <- test_data$lwage

# For LASSO
set.seed(123) # For reproducibility
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)

# For Ridge Regression
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Predictions for LASSO
predictions_lasso <- predict(cv_lasso, newx = x_test, s = "lambda.min")

# Predictions for Ridge
predictions_ridge <- predict(cv_ridge, newx = x_test, s = "lambda.min")

# Calculate MSE
mse_lasso <- mean((predictions_lasso - y_test)^2)
mse_ridge <- mean((predictions_ridge - y_test)^2)

print(paste("LASSO MSE:", mse_lasso))
print(paste("Ridge MSE:", mse_ridge))



## g)


# Fit the GAM
gam_model <- gam(lwage ~ s(ed) + sex_numeric, data = train_data)

# Summary of the model
summary(gam_model)

# Make predictions on the test set
predictions_gam <- predict(gam_model, newdata = test_data)

# Calculate MSE
mse_gam <- mean((predictions_gam - test_data$lwage)^2)

print(paste("GAM MSE:", mse_gam))


## h)

```{r, eval=T, error=TRUE, message=TRUE}

# Using 'lwage' as the response variable and 'sex_numeric' and 'ed' as predictors
tree_model <- rpart(lwage ~ sex_numeric + ed, data=train_data, method="anova")


rpart.plot(tree_model, type=4, extra=101)

# Making predictions on the test set
predictions_tree <- predict(tree_model, newdata=test_data)

# Evaluate the predictions, e.g., using Mean Squared Error (MSE)
mse_tree <- mean((predictions_tree - test_data$lwage)^2)
print(paste("MSE for the tree:", mse_tree))

# Prune the tree
pruned_tree <- prune(tree_model, cp=tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"])

# Plot the pruned tree
rpart.plot(pruned_tree, type=4, extra=101)

# Evaluate the pruned tree's predictions
predictions_pruned <- predict(pruned_tree, newdata=test_data)
mse_pruned <- mean((predictions_pruned - test_data$lwage)^2)
print(paste("MSE for the pruned tree:", mse_pruned))

