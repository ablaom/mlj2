using MLJ

X, y = datanow(); # ALL of the data, training, test and validation

# split the rows into training and testing rows:
fold1, fold2 = partition(eachindex(y), 0.7) # 70:30 split


## Cross-validation the hard way

# construct a transformer to standardize the inputs, using the
# training fold to prevent data leakage:
scale_ = Standardizer() 
scale = prefit(scale_, X) 
fit!(scale, fold1)

# get the transformed inputs:
Xt = transform(scale, X)

# convert data frame `Xt` to an array:
Xa = array(Xt)

# choose a learner and train it on the same fold:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
knn = prefit(knn_, Xa, y) 
fit!(knn, fold1)

# get the predictions on the other fold:
yhat = predict(knn, Xa(fold2));

# compute the error:
er1 = rms(y(fold2), yhat)

# then repeat all of the above with roles of `fold1` and `fold2`
# interchanged to get `er2`.

# average to get estimate of
# generalization error.


## Cross-validation using dynamic data:

# we need two lines of code not used above but everything else will be
# easier, but with practically identical syntax:

X = dynamic(X)
y = dynamic(y)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
scale = prefit(scale_, X) # no need to train!

# get the transformed inputs, as if `scale` were already trained:
Xt = transform(scale, X)

# convert DataFrame Xt to an array:
Xa = array(Xt)

# choose a learner and make it trainable:
knn_ = KNNRegressor(K=7) 
knn = prefit(knn_, Xa, y) # no need to train!

# get the predictions, as if `knn` already trained:
yhat = predict(knn, Xa)

# compute the error:
er = rms(y, yhat)

# Now `er` is dynamic, so we can do "look through training" on any rows
# we like and evaluate on any rows:
fit!(er, fold1)
er1 = er(fold2)

fit!(er, fold2)
er2 = er(fold1)

er = (er1 + er2)/2

