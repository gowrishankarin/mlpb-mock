options(scipen = 20, digits = 4)

lapply(c("e1071", "tm"), require, character.only=T)
# e1071 for Naive Bayes
# tm for counting word frequencies

#========================================================
# Load data

jobtitles <- read.csv("_data/jobtitles.csv", na.strings = c("NA", ""))

#========================================================
# Count word frequencies
# Here we make use of the tm (text mining) package


# first build a Vector Corpus Object
my.corpus <- VCorpus(VectorSource(jobtitles$job_title))

# now build a document term matrix
dtm <- DocumentTermMatrix(my.corpus)

# inspect the results
inspect(dtm)

#=======================================================
# Train a naive bayes model

# Put word frequencies into a data.frame and convert column types from numeric to factor
# (so naiveBayes() knows the xi is a Bernoulli random variable not Gaussian)

# Prepare training data
train.x <- data.frame(inspect(dtm)[1:10, ]) # Use first 10 samples to build the training set
train.x[, 1:10] <- lapply(test.x, FUN=function(x) {
    factor(x, levels = c("0", "1"))
})

train.y <- factor(jobtitles$job_category[1:10])

# Prepare test data
test.x <- data.frame(inspect(dtm)[11:12,]) # Use the last 2 samples to build the test set
test.x[, 1:10] <- lapply(test.x, FUN=function(x) {
    factor(x, levels = c("0", "1"))
})

# train model
classifier <- naiveBayes(x = train.x, y = train.y, laplace = 0.000000001) # use laplace of nearly 0 alpha value

# make prediction on the unlabeled data
predict(classifier, test.x, type="raw")

#========================================================
# Train a naive bayes modelwith laplace = 1

# train model
classifier <- naiveBayes(x=train.x, y=train.y, laplace=1) # use laplace of 1

# Make prediction on the unlabeled data
predict(classifier, test.x, type="raw")


























