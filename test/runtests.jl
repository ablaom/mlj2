using Revise
using Test
using MLJ

@constant junk=KNNRegressor()

# we first test KNN as it will be used to check other functinality:
Xt = [4 2 5 3;
     2 1 6 0.0]

@test MLJ.KNN.distances_and_indices_of_closest(3,
        MLJ.KNN.euclidean, Xt, [1, 1])[2] == [2, 4, 1]

X = Xt' |> collect
y = Float64[2, 1, 3, 8]
knn_ = KNNRegressor(K=3)
allrows = 1:4
estimator, state, report = fit(knn_, X, y, nothing, 0); # fit(model, X, y, state, verbosity)
@test report == nothing
@test estimator == state

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = [1.0 1.0]
yhat = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test isapprox(predict(knn_, estimator, Xtest)[1], yhat)


## MODEL INTERFACE

X_frame, y = datanow(); # boston data
X = array(X_frame)

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y)
train, valid, test = partition(allrows, 0.7, 0.15)
@test vcat(train, valid, test) == allrows
knn = prefit(knn_, X, y)
fit!(knn, train);
er = rms(predict(knn, X(test)), y(test))

# TODO: compare to constant regressor and check it's significantly better


## DYNAMIC DATA INTERFACE

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn))

X, y = datanow(); # ALL of the data, training, test and validation

# split the rows into training and testing rows:
fold1, fold2 = partition(eachindex(y), 0.7) # 70:30 split

X = dynamic(X)
y = dynamic(y)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
@constant scale = prefit(scale_, X) # no need to fit

# get the transformed inputs, as if `scale` were already fit:
Xt = transform(scale, X)

# convert DataFrame Xt to an array:
Xa = array(Xt)

# choose a learner and make it trainable:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
knn = prefit(knn_, Xa, y) # no need to fit

# get the predictions, as if `knn` already fit:
yhat = predict(knn, Xa)

# compute the error:
er = rms(y, yhat)

# Now `er` is dynamic, so we can do "look through training" on any rows
# we like and evaluate on any rows:
fit!(er, fold1)
er(fold2)

fit!(er, fold2)
er(fold1)

include("Transformer.jl")
include("DecisionTree.jl")


