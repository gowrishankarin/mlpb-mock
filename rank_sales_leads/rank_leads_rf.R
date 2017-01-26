# Random Forest model where we first reduce the cardinality of each categorical feature
# by basketing infrequent values int "catch-all" groups

options(scipen = 20, digits = 4)

lapply(c(
    "data.table",
    "stringr",
    "randomForest",
    "pROC"
), require, character.only=T)

train <- fread("_data/train.csv")
test <- fread("_data/test.csv")

# View 
train
# Some thoughts and ideas come to mind
# - CompanyName could be useful in theory, but we are not goit to deal with the complexities of 
# NLP featture engineering here
# - Phone number probably isn't useful... but maybe AreaCode is since it should separate the samples
# by geography => demographics
# - It probably makes sense to order the Contact values: general line < other < manager < owner
# (this affects the model's construction)
# - Website (like PhoneNumber) is too unique to be usefulm but maybe businesses that use .net and .org 
# are less likely to buy our software

# Now let's see what our overall hit ratio is
train[, list(Samples = .N, Sales=sum(Sale), HitRatio = sum(Sale)/.N)]

#==========================================================================
# Feature engineering and tranforming the training dataset

# Some things to keep in mind
# - We have to deal with missing values(NAs). We could impute mean or median into their place,
# but this is likely to degrade the model's perfection. NAs here have special meaning. eg. FacebookLikes = NA
# means the company does not have a facebook
# This has very different implications than, "The company has a facebook, but we didnt bother to track how
# many likes it has"

# - RandomeFOrest only accepts numeric and factor values and it can't handle NAs. So we need to convert
# categorical features to factors and change NAs to some specific value, eg. "NA_Val", or -1 for numeric 
# features. Also need to convert Sale from logical to factor

# - RandomForest can only handle factors with up to 53 unique levels. We won't run into this issue since
# out dataset is small, but for illustration we'll create a "catch all" group for TypeOfBusiness to use for
# uncommon business types

# - The test set could have a TypeOfBusiness not seen in the training set. Need to prepare for that

#-------------------------------------------------
# Type of Business

# Generate a map to map the values in TypeOfBusiness to a new field, TOBCategory
tobMap <- train[, list(Samples = .N), by = TypeOfBusiness]
tobMap[, TOBCategory := ifelse(Samples < 2, "Other_Val", TypeOfBusiness)] # Map all uncommon categories to "Other_Val"
tobMap[is.na(TypeOfBusiness), TOBCategory := "NA_Val"] # Map NA categories to "NA_Val"
tobMap[, TOBCategory := factor(TOBCategory)] # convert to factor

# train
train[tobMap, TOBCategory := TOBCategory, on = "TypeOfBusiness"]

# test
test[tobMap, TOBCategory := TOBCategory, on = "TypeOfBusiness"]
test[is.na(TOBCategory), TOBCategory := "Other_Val"] # NAs will be generated for brand new categories. Send these to "Other_Val"

#-------------------------------------------------
# Contact

# In this case, we know all the possible contact types
contacts <- c("general line", "other", "manager", "owner") # Note the order of the elements
train[, Contact := factor(Contact, ordered=TRUE)] # IMPORTANT Note the ordered = TRUE bit
test[, Contact := factor(Contact, ordered = TRUE)]

#-------------------------------------------------
# AreaCode

train[, AreaCode := substr(PhoneNumber, 1, 3)] # train
test[, AreaCode := substr(PhoneNumber, 1, 3)] # test

# We need to convert AreaCode to type factor
areacodes <- unique(c(train$AreaCode, test$AreaCode)) # Get all the unique area codes
train[, AreaCode := factor(AreaCode, levels=areacodes)] # train
test[, AreaCode := factor(AreaCode, levels = areacodes)] # test

#------------------------------------------------
# Website Extension (note: str_detect comes from the stringr package)

extensions <- c("com", "net", "org", "other", "none")

# train
train[is.na(Website), WebsiteExtension := "none"]
train[str_detect(Website, ".com"), WebsiteExtension := "com"]
train[str_detect(Website, ".net"), WebsiteExtension := "net"]
train[str_detect(Website, ".org"), WebsiteExtension := "org"]
train[is.na(WebsiteExtension), WebsiteExtension := "other"]
train[, WebsiteExtension := factor(WebsiteExtension, levels = extensions)]

# test
test[is.na(Website), WebsiteExtension := "none"]
test[str_detect(Website, ".com"), WebsiteExtension := "com"]
test[str_detect(Website, ".net"), WebsiteExtension := "net"]
test[str_detect(Website, ".org"), WebsiteExtension := "org"]
test[is.na(WebsiteExtension), WebsiteExtension := "other"]
test[, WebsiteExtension := factor(WebsiteExtension, levels = extensions)]

#-----------------------------------------------
# FacebookLikes

# Note -1L is type integer vs -1 which is type double
train[is.na(FacebookLikes), FacebookLikes := -1L] # train
test[is.na(FacebookLikes), FacebookLikes := -1L] # test

#------------------------------------------------
# Twitter Followers
train[is.na(TwitterFollowers), TwitterFollowers := -1L] # train
test[is.na(TwitterFollowers), TwitterFollowers := -1L] # test

#-----------------------------------------------
# Sale (the target variable)

# Create a separate feature called Target. Then we can still
# do stuff like train[, list(Sales=sum(Sales))]

train[, Target := factor(Sale, levels = c("FALSE", "TRUE"))]

#===============================================
# Random Forest Model

features <- c("TOBCategory", "Contact", "AreaCode", "WebsiteExtension", "FacebookLikes", "TwitterFollowers")
rf <- randomForest(
    x=train[, features, with=FALSE],
    y=train$Target,
    ntree = 200,
    mtry = 2,
    nodesize = 2
)

importance(rf)

#===============================================
# MAke some predictions on the test set & evaluate the results

# Make Predictions
test[, ProbSale := predict(
    rf, newdata = test[, features, with=FALSE], type="prob"
)[, 2]]

#----------------------------------------------
# Rank the predictions from most likely to least likely

setorder(test, -ProbSale)
test[, ProbSaleRk := .I]

# Take a look
test[, list(ProbSaleRk, CompanyName, ProbSale, Sale)] # Looks pretty good

#----------------------------------------------
# Let's evaluate the results using area under the ROC curve using the pROC package

rocCurve <- roc(response=test$Sale, predictor=test$ProbSale, direction="<")
rocCurve$auc
plot(rocCurve)










