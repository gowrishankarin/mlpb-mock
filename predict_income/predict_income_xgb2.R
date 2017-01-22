options(scipen = 20)

lapply(c(
	"data.table",
	"ggplot2",
	"xgboost",
	"DiagrammeR",
	"Matrix"
), require, character.only=T)

# Root Mean Squared Error
rmse <- function(preds, actuals) {
    sqrt(mean((preds - actuals)^2)) # |a - b|
}

train <- fread("_data/train.csv")
test <- fread("_data/test.csv")
setnames(test, "Income", "IncomeTruth") # Change col name Income to IncomeTruth

#==========================================================================================
# Build modified training dataset

dt <- rbind(
	train[, list(CountryID, RegionID, CityID, Income)],
	test[, list(CountryID, RegionID, CityID, Income=NA)]
)

countries <- dt[, list(.N, AvgIncome=mean(Income, na.rm=TRUE)), key=CountryID]
countries[, CountryIdx := .I]
regions <- dt[, list(.N, AvgIncome=mean(Income, na.rm=TRUE)), key=RegionID]
regions[, RegionIdx := .I]
cities <- dt[, list(.N, AvgIncome=mean(Income, na.rm=TRUE)), key=CityID]
cities[, CityIdx := .I]

train[countries, CountryIdx := CountryIdx, on="CountryID"]
train[regions, RegionIdx := RegionIdx, on="RegionID"]
train[cities, CityIdx := CityIdx, on="CityID"]

test[countries, CountryIdx := CountryIdx, on="CountryID"]
test[regions, RegionIdx := RegionIdx, on="RegionID"]
test[cities, CityIdx := CityIdx, on="CityID"]

train[, Idx := .I]
test[, Idx := .I]

train_countryM <- sparseMatrix(i=train$Idx, j=train$CountryIdx, x=1)
train_regionM <- sparseMatrix(i=train$Idx, j=train$RegionIdx, x=1)
train_cityM <- sparseMatrix(i=train$Idx, j=train$CityIdx, x=1)

test_countryM <- sparseMatrix(i=test$Idx, j=test$CountryIdx, x=1)
test_regionM <- sparseMatrix(i=test$Idx, j=test$RegionIdx, x=1)
test_cityM <- sparseMatrix(i=test$Idx, j=test$CityIdx, x=1)

trainM <- do.call(cBind, list(train_countryM, train_regionM, train_cityM))
testM <- do.call(cBind, list(test_countryM, test_regionM, test_cityM))

#==========================================================================================
# XGBoost that puppy

features <- c(
	paste0("Country", countries$CountryID), 
	paste0("Region", regions$RegionID), 
	paste0("City", cities$CityID)
)

#------------------------------------------------------------------------------

paramList <- list(
	eta=0.2, 
	gamma=0, 
	max.depth=3, 
	min_child_weight=1, 
	subsample=0.9, 
	colsample_bytree=1
)

bst.cv <- xgb.cv(
	params=paramList,
	data=trainM,
	label=as.matrix(train$Income),
	nfold=5,
	early_stop_round=10,
	eval_metric="rmse",
	nrounds=200,
	prediction=TRUE
)

bst <- xgboost(
	params=paramList,
	data=trainM,
	label=as.matrix(train$Income),
	nrounds=190 # nrow(bst.cv[["dt"]])-10
)

#===================================================================================
# Predict and Evaluate

# Predict
train[, IncomeXGB := predict(bst, trainM)]
test[, IncomeXGB := predict(bst, testM)]

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














