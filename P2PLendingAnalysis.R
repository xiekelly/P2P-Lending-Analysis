# Kelly Xie
# April 18
# Prof. Tambe, Analytics in the Digital Economy
# Lab 5: Peer-to-peer lending


################ 1. Data cleanup ################


# removes first row
train.loans = read.csv("/data/LoanStats3c.csv", skip=1)
test.loans = read.csv("/data/LoanStats3d.csv", skip=1)

# removes last 2 rows
train.loans = train.loans[-((nrow(train.loans)-1):nrow(train.loans)),]
test.loans = test.loans[-((nrow(test.loans)-1):nrow(test.loans)),]


################# 2. Descriptive statistics ################


# creates binary variable for grade
train.loans$highgrade = (train.loans$grade == "A") | (train.loans$grade == "B")

# proportion of loans that received highgrade
percent.highgrade = mean(train.loans$highgrade)*100

# t-test results for differences in proportion of highgrade:
# whether the debtor is above or below the median income level
median.income = median(train.loans$annual_inc)
train.loans$annual_inc_above_median = train.loans$annual_inc >= median.income
t.test(highgrade~annual_inc_above_median, data = train.loans)

# whether the loan request is above or below the median loan amount
median.loan = median(train.loans$loan_amnt)
train.loans$loan_above_median = train.loans$loan_amnt >= median.loan
t.test(highgrade~loan_above_median, data = train.loans)

# whether the debtor rents their home or not
train.loans$home_rented = train.loans$home_ownership == "RENT"
t.test(highgrade~home_rented, data = train.loans)


################# 3. Logical classifier on training data ################


# performs a logistic regression that attempts to predict highgrade 
# using annual income, home ownership, and loan amount as the predictors
model = glm(highgrade ~ annual_inc + home_ownership + loan_amnt, 
            data = train.loans, family = binomial)
summary(model)

# generates a vector of the probabilities that are predicted by the logistic regression
train.loans$predict_val = predict(model, type="response")

# creates a new variable that classifies loans as being highgrade or not, 
# based on predicted probabilities
train.loans$predict_highgrade = train.loans$predict_val > 0.5 # probability threshold

# Evaluates how well this logistic regression-based classifier performs
# where accuracy is the proportion of rows in which the classifer prediction 
# is equal to its actual highgrade value.

# Benchmarks:
# 1. What is the accuracy of this classifer on the training data?
mean(train.loans$predict_highgrade == train.loans$highgrade)

# 2. What would be the accuracy of a classifier that randomly assigns 
# 0 and 1 values as the predicted class?
train.loans$predict_highgrade_random = runif(nrow(train.loans)) > 0.5 # assign random
mean(train.loans$predict_highgrade_random == train.loans$highgrade)

# 3. What is the accuracy of a classifier that simply assigns a 
# value of 0 to all rows for the predicted class?
train.loans$predict_highgrade_zero = 0 # assign 0s
mean(train.loans$predict_highgrade_zero == train.loans$highgrade)


################# 4. Supervised learning ################


library(rpart)
require(rpart)

# builds a classification tree
fit = rpart(highgrade ~ annual_inc + home_ownership + loan_amnt, 
            data = train.loans, method = "class")
plot(fit)
text(fit)

# predicts values using classification tree
train.loans$predict_highgrade_tree = predict(fit, type="class")

# calculates accuracy
mean(train.loans$predict_highgrade_tree == train.loans$highgrade)

# Machine learning based classifier is more accurate than logistic regression model
# in predicting values.


################# 5. Model performance on test data ################


# Evaluates the accuracy of both of the classifiers on the test data.
test.loans$highgrade = (test.loans$grade == "A") | (test.loans$grade == "B")

# 1. logistic regression classifier
test.loans$predict_val = predict(model, newdata=test.loans, type="response")
test.loans$predict_highgrade_reg = test.loans$predict_val > 0.5 # probability threshold
mean(test.loans$predict_highgrade_reg == test.loans$highgrade)

# 2. machine learning classifier
test.loans$predict_highgrade_tree = predict(fit, newdata=test.loans, type="class")
mean(test.loans$predict_highgrade_tree == test.loans$highgrade)

# As a benchmark, what is the accuracy of a classifier that randomly 
# assigns 0 and 1 values to the test data?
test.loans$predict_highgrade_random = runif(nrow(test.loans)) > 0.5 # assign random
mean(test.loans$predict_highgrade_random == test.loans$highgrade)

# As another benchmark, what is the accuracy of a classifier that 
# simply assigns a value of 0 to all rows of the test data?
test.loans$predict_highgrade_zero = 0 # assign 0s
mean(test.loans$predict_highgrade_zero == test.loans$highgrade)

