# Gradient Boosting Model using XGBoost

# Notes about this model.
# XGBoost requires a single matrix of numeric values as input. This means non ordered categorical features
# need to be one-hot-encoded. This can result in a really large sparse matrix but XGBoost allows for the input 
# martrix to be represented in compressed format via dgCMatrix, we will utilize that option here

options(scipen = 20, digits = 4)

lapply(c(
    "data.table",
    "stringr",
    "Matrix",
    "xgboost",
    "pROC"
), require, character.only=T)

train <- fread("_data/train.csv")
test <- fread("_data/test.csv")

#==============================================
# Really quick and dirty analysis

train

train[, list(Samples=.N, Sales=sum(Sale), HitRatio=sum(Sale)/.N)]

#==============================================
# Feature engineering and transforming the training dataset

# Some things to keep in mind
# - Need to build a sparse matrix for each non-ordered categorical feature: TypeOfBusiness, AreaCode and WebsiteExtension
# - Need to concatenate all features into one big sparse matrix
# - XGBoost has smart handling for NA Values, so we dont need to impute values for NA in FacebookLikes and TwitterFollowers

#----------------------------------------------
# Create some row indexes to help us populate sparse matricies

train[, RowIdx := .I]
test[, RowIdx := .I]

tobMap <- train[, list(Samples=.N), by=TypeOfBusiness]
tobMap[, TobIdx := .I]

#train
train[tobMap, TobIdx := TobIdx, on = "TypeOfBusiness"]
trainTOBSparseM <- sparseMatrix(
    i=train$RowIdx,
    j=train$TobIdx,
    x=1,
    dimnames=list(NULL, tobMap$TypeOfBusiness)
) # Build a sparse matrix

# test
test[tobMap, TobIdx := TobIdx, on="TypeOfBusiness"]
testTOBSparseM <- sparseMatrix(
    i=test[!is.na(TobIdx)]$RowIdx, 
    j=test[!is.na(TobIdx)]$TobIdx,
    x = 1,
    dims = c(nrow(test), nrow(tobMap)),
    dimnames = list(NULL, tobMap$TypeOfBusiness)
)

#----------------------------------------------------------
# Area Code

train[, AreaCode := substr(PhoneNumber, 1, 3)] # train
test[, AreaCode := substr(PhoneNumber, 1, 3)] # test

# Generate a map to map the values in AreaCode to a sparse matrix
areacodeMap <- train[, list(Samples=.N), by=AreaCode]
areacodeMap[, AreaCodeIdx := .I]

# train
train[areacodeMap, AreaCodeIdx := AreaCodeIdx, on="AreaCode"]
trainAreaCodeSparseM <- sparseMatrix(
    i=train$RowIdx,
    j=train$AreaCodeIdx,
    x=1, 
    dimnames=list(NULL, areacodeMap$AreaCode)
)

# test
test[areacodeMap, AreaCodeIdx := AreaCodeIdx, on="AreaCode"]
testAreaCodeSparseM <- sparseMatrix(
    i=test[!is.na(AreaCodeIdx)]$RowIdx,
    j=test[!is.na(AreaCodeIdx)]$AreaCodeIdx,
    x=1,
    dims=c(nrow(test), nrow(areacodeMap)),
    dimnames = list(NULL, areacodeMap$AreaCode)
)

#--------------------------------------------------------
# Website Extension 

extensions <- c("com", "net", "org", "other", "none")

# train
train[is.na(Website), WebsiteExtension := "none"]
train[str_detect(Website, ".com"), WebsiteExtension := "com"]
train[str_detect(Website, ".net"), WebsiteExtension := "net"]
train[str_detect(Website, ".org"), WebsiteExtension := "org"]
train[is.na(WebsiteExtension), WebsiteExtension := "other"]

# test
test[is.na(Website), WebsiteExtension := "none"]
test[str_detect(Website, ".com"), WebsiteExtension := "com"]
test[str_detect(Website, ".net"), WebsiteExtension := "net"]
test[str_detect(Website, ".org"), WebsiteExtension := "org"]
test[is.na(WebsiteExtension), WebsiteExtension := "other"]

# Generate a map to map the values in WebsiteExtension to a sparse matrix
extensionMap <- train[, list(Samples=.N), by=WebsiteExtension]
extensionMap[, ExtensionIdx := .I]

# train 
train[extensionMap, ExtensionIdx := ExtensionIdx, on = "WebsiteExtension"]
trainExtensionSparseM <- sparseMatrix(
    i=train$RowIdx,
    j=train$ExtensionIdx,
    x=1,
    dimnames=list(NULL, extensionMap$WebsiteExtension)
)

# test
test[extensionMap, ExtensionIdx := ExtensionIdx, on="WebsiteExtension"]
testExtensionSparseM <- sparseMatrix(
    i=test[!is.na(ExtensionIdx)]$RowIdx,
    j=test[!is.na(ExtensionIdx)]$ExtensionIdx,
    x=1,
    dims=c(nrow(test), nrow(extensionMap)),
    dimnames=list(NULL, extensionMap$WebsiteExtension)
)

#-------------------------------------------
# Contact (convert to numeric, 1-4)

# In this case, we know all the possible contact types
contacts <- c("general line", "other", "manager", "owner")
train[, Contact := match(Contact, contacts)]
test[, Contact := match(Contact, contacts)]

#----------------------------------------------
# Combine training features into a single sparse matrix

# Insert the non sparse features into a dgCMatrix
trainNonSparseFeats <- Matrix(
    as.matrix(train[, list(Contact, FacebookLikes, TwitterFollowers)]), 
    sparse=TRUE
)
testNonSparseFeats <- Matrix(
    as.matrix(test[, list(Contact, FacebookLikes, TwitterFollowers)]),
    sparse=TRUE
)

# Combine all the sparse matrices into one big sparse matrix
trainM <- do.call(cBind, list(trainNonSparseFeats, trainTOBSparseM, trainAreaCodeSparseM, trainExtensionSparseM))
testM <- do.call(cBind, list(testNonSparseFeats, testTOBSparseM, testAreaCodeSparseM, testExtensionSparseM))

# Extract the column names of the matrix into a vector called 'features'
features <- c(dimnames(trainM[[2]]))

#===================================================
# XGBoost Model

set.seed(2016)
boostingParams = list(
    objective="binary:logistic", 
    eval_metric="auc", 
    eta=0.3,
    max.depth=10,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=1,
    gamma=0,
    lambda=0,
    alpha=0
)
bst <- xgboost(params=boostingParams, data=trainM, label=as.matrix(train$Sale)*1, nrounds=5)

#----------------------------------------------------
# Check the feature importance

xgb.importance(bst, featuer_names=features)

#=====================================================
# Make some predictions on the test set & evaluage the results

# Make Predictions
test[, ProbSale := predict(bst, newdata=testM)]

#-----------------------------------------------------
# Rank the predictions from most likely to least likely

setorder(test, -ProbSale)
test[, ProbSaleRk := .I]

#-----------------------------------------------------
# Take a look

test[, list(ProbSaleRk, CompanyName, ProbSale, Sale)]


#------------------------------------------------------
# Let's evaluage the results using area under the ROC curve using the pROC package

rocCurve <- roc(response=test$Sale, predictor = test$ProbSale, direction = "<")
rocCurve$auc
plot(rocCurve)










