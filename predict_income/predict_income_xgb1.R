options(scipen = 20)

lapply(c("data.table", "ggplot2", "xgboost", "DiagrammeR"),
       require, character.only=T)

# Root Mean Squared Error
rmse <- function(preds, actuals) {
    sqrt(mean((preds - actuals)^2)) # |a - b|
}

chunk <- function(x, n) {
    # split a vector into a list of vectors of equal size(or nearly equal size)
    split(x, cut(seq_along(x), n, labels = FALSE))
}

train <- fread("_data/train.csv")
test <- fread("_data/test.csv")
setnames(test, "Income", "IncomeTruth") # Change col name Income to IncomeTruth

transformTrain <- function(folds = 5) {
    # Splits the training set into disjoint (train, test) pairs: {(train1, test1), (train2, test2), ...} 
    # for number of specified folds
    # For a given (train_k, test_k) pair, the incomes in train_k are averaged by City, Region and 
    # Country(separately) and then inserted in to test_k appropriately
    # Finally the test sets are concatenated, producing a new training dataset
    
    test_folds <- chunk(sample(nrow(train), nrow(train)), folds)
    train_folds <- lapply(test_folds, function(testIdxs) 
        seq(nrow(train))[-testIdxs])
    
    
    tests <- lapply(seq_len(folds), FUN = function(i) {
        train1 <- train[train_folds[[i]]]
        train1_countries <- train1[, list(Countries=.N, CountryAvg=mean(Income)), by=list(CountryID)]
        train1_regions <- train1[, list(Regions=.N, RegionAvg=mean(Income)), by=list(RegionID)]
        train1_cities <- train1[, list(Cities=.N, CityAvg=mean(Income)), by=list(CityID)]
        test1 <- train[test_folds[[i]]]
        test1 <- train1_countries[test1, on="CountryID"]
        test1 <- train1_regions[test1, on="RegionID"]
        test1 <- train1_cities[test1, on="CityID"]
        
        return(test1)
    })
    
    # Build the new training dataset by concatenating all the test sets
    train_new <- rbindlist(tests, use.names = TRUE)
    
    # Return a list of the trainIdxs, testIdxs and the new training dataset
    return(list(trainIdxs=train_folds, testIdxs=test_folds, train=train_new))
}

# Create the new training set
transformed <- transformTrain(5)
train_new <- transformed[["train"]]
trainIdxs <- transformed[["trainIdxs"]]
testIdxs <- transformed[["testIdxs"]]

# Create the modified test set
countryAvgs <- train[, list(Countries=.N, CountryAvg=mean(Income)), keyby="CountryID"]
regionAvgs <- train[, list(Regions=.N, RegionAvg=mean(Income)), keyby="RegionID"]
cityAvgs <- train[, list(Cities=.N, CityAvg=mean(Income)), keyby="CityID"]

test <- countryAvgs[test, on="CountryID"]
test <- regionAvgs[test, on="RegionID"]
test <- cityAvgs[test, on="CityID"]

#===================================================================================
# XGBoost that puppy

features <- c("Cities", "CityAvg", "Regions", "RegionAvg", "Countries", "CountryAvg")

# Test various hyperparamenters and values here and see what works best. (A poor man's grid search)
paramList <- list(
    eta =0.2,
    gamma = 0,
    max.depth = 3,
    min_child_weight = 1,
    subsample = 0.9,
    colsample_bytree = 1
)
bst.cv <- xgb.cv(
    params=paramList,
    data=as.matrix(train_new[, features, with=FALSE]),
    label=as.matrix(train_new$Income),
    folds=testIdxs,
    early_stop_round=3,
    eval_metric="rmse",
    nrounds = 200,
    prediction = TRUE
)
bst <- xgboost(
    params=paramList,
    data=as.matrix(train_new[, features, with=FALSE]),
    label=as.matrix(train_new$Income),
    nrounds = 200
)

#===================================================================================
# Predict and Evaluate

# Predict
train[, IncomeXGB := predict(bst, as.matrix(train_new[, features, with=FALSE]))]
test[, IncomeXGB := predict(bst, as.matrix(test[, features, with=FALSE]))]

# Trees
bst.trees <- xgb.model.dt.tree(features, model=bst)
bst.trees[Tree==0]

# Importance
xgb.importance(model=bst, features)


#------------------------------------------------------------------------------------
# Importance
bst.minimal <- xgboost(
    params = paramList,
    data = as.matrix(train_new[, features, with=FALSE]),
    label = as.matrix(train_new$Income),
    nrounds = 3
)
xgb.plot.tree(features, model = bst.minimal)


#-----------------------------------------------------------------------------------
# Evaluate

rmse(train$IncomeXGB, train$Income)
rmse(test$IncomeXGB, test$IncomeTruth)

# Errors
train[, SE := (IncomeXGB-Income)^2]
test[, SE := (IncomeXGB-IncomeTruth)^2]
test[order(SE)]



















