options(scipen=10)

lapply(c(
  "data.table",
  "ggplot2",
  "mltools", # For generating CV folds and one hot encoding
  "class", # K-Nearest Neighbhours Model
  "LiblineaR" # SVM and Logistic Regression
), require, character.only=T)

train <- fread("_data/train.csv")
test <- fread("_data/test.csv")

#----------------------------------------
# Fix column types

train[, Competitor := factor(Competitor)]
test[, Competitor := factor(Competitor)]

#----------------------------------------
# Add feature DistFromCenter

train[, DistFromCenter := sqrt(XCoord^2 + YCoord^2)]
test[, DistFromCenter := sqrt(XCoord^2 + YCoord^2)]

#----------------------------------------
# Build folds for cross validation and stacking

train[, FoldID := folds(Competitor, nfolds = 5, stratified = TRUE, seed = 2016)] # mltools function

#=========================================
# KNN
#
# Do a grid search for k = 1, 2, ... 30 by cross validating model using folds 1-5
# ie [test=f1, train=(f2, f3, f4, f5)], [test=f2, train=(f1, f3, f4, f5)], ...

knnCV <- list()
knnCV[["Features"]] <- c("XCoord", "YCoord")
knnCV[["ParamGrid"]] <- CJ(k=seq(1, 30))
knnCV[["BestScore"]] <- 0

# Loop through each set of parameters
for(i in seq_len(nrow(knnCV[["ParamGrid"]]))) {
    
    # Get the ith set of parameters
    params <- knnCV[["ParamGrid"]][i]
    
    # Build an empty vector to store scores from each train/test fold
    scores <- numeric()
    
    # Build an empty list to store predictions from each train/test fold
    predsList <- list()
    
    # Loop through each test fold, fit model to training folds and make predictions on test fold
    for(foldID in 1:5) {
        # Build the train/test folds
        testFold <- train[J(FoldID = foldID), on="FoldID"]
        trainFolds <- train[!J(FoldID = foldID), on="FoldID"] # Exclude fold i from trainFolds
        
        # Train the model and make predictions
        testFold[, Pred := knn(train = trainFolds[, knnCV$Features, with=FALSE], test=testFold[, knnCV$Features, with=FALSE], cl=trainFolds$Competitor, k=params$k)]
        predsList <- c(predsList, list(testFold[, list(ID, FoldID, Pred)]))
        
        # Evaluate predictions (accuracy rate) and append score to scores vector
        score <- mean(testFold$Pred == testFold$Competitor)
        scores <- c(scores, score)
    }
    
    # Measure the overall score. If best, tell knnCV
    score <- mean(scores)
    
    # Insert the score into ParamGrid
    knnCV[["ParamGrid"]][i, Score := score][]
    print(paste("Params:", paste(colnames(knnCV[["ParamGrid"]][i]), knnCV[["ParamGrid"]][i], collapse = " | ")))
    
    if(score > knnCV[["BestScore"]]) {
        knnCV[["BestScores"]] <- scores
        knnCV[["BestScore"]] <- score
        knnCV[["BestParams"]] <- knnCV[["ParamGrid"]][i]
        knnCV[["BestPreds"]] <- rbindlist(predsList)
    }
}

# Check the best parameters
knnCV[["BestParams"]]

# Plot the score for each k value 
knnCV[["ParamGrid"]]
ggplot(knnCV[["ParamGrid"]], aes(x=k, y=Score)) + geom_line() + geom_point()

#=========================================
# SVM
#
# Do a grid search for k = 1, 2, ... 30 by cross validating model using folds 1-5
# ie [test=f1, train=(f2, f2, f4, f5)], [test=f2, train=(f1, f3, f4, f5)], ...

svmCV <- list()
svmCV[["Features"]] <- c("XCoord", "YCoord", "DistFromCenter")
svmCV[["ParamGrid"]] <- CJ(type=1:5, cost=c(0.01, 0.1, 1, 10, 100, 1000, 2000), Score=NA_real_)
svmCV[["BestScore"]] <- 0

# Loop through eahc set of parameters
for(i in seq_len(nrow(svmCV[["ParamGrid"]]))) {
    
    # Get the ith set of parameters
    params <- svmCV[["ParamGrid"]][i]
    # Build an empty vector to stroe scores from each train/test fold
    scores <- numeric()
    
    # Buidl an empty list ot store predictions frome each train/test fold
    predsList <- list()
    
    # Loop through each test fold, fit model to training folds and make predictions on test fold
    
    for(foldID in 1:5) {
        # Build the train/test folds
        testFold <- train[J(FoldID=foldID), on="FoldID"]
        trainFolds <- train[!J(FoldID = foldID), on="FoldID"] # Exclude fold i from trainFolds
        
        # Train the model and make predictions
        svm <- LiblineaR(
            data = trainFolds[, svmCV$Features, with=FALSE],
            target = trainFolds$Competitor,
            type = params$type,
            cost = params$cost
        )
        
        testFold[, Pred := predict(svm, testFold[, svmCV$Features, with=FALSE])$predictions]
        predsList <- c(predsList, list(testFold[, list(ID, FoldID, Pred)]))
        
        # Evaluate predictions and append score to scores vector
        score <- mean(testFold$Pred == testFold$Competitor)
        scores <- c(scores, score)
    }
    
    # Measure the overall score. If best, tell svmCV
    score <- mean(scores)
    
    # Insert the score into ParamGrid
    svmCV[["ParamGrid"]][i, Score := score][]
    print(paste("Params:", paste(colnames(svmCV[["ParamGrid"]][i]), svmCV[["ParamGrid"]][i], collapse = " | ")))
    
    if(score > svmCV[["BestScore"]]) {
        svmCV[["BestScores"]] <- scores
        svmCV[["BestScore"]] <- score
        svmCV[["BestParams"]] <- svmCV[["ParamGrid"]][i]
        svmCV[["BestPreds"]] <- rbindlist(predsList)
    }
}

# Check the best params
svmCV[["BestParams"]]

# Plot the score for each (cost, type) pairs
svmCV[["ParamGrid"]]
ggplot(svmCV[["ParamGrid"]], aes(x=cost, y=Score, color=factor(type))) + geom_line() + geom_point()






















