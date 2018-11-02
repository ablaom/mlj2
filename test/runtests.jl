using Revise
using Test
using MLJ

## DATA ADAPTOR

import DataFrames: DataFrame, names

df = DataFrame(x=[1,2,3], y=["a", "b", "c"]);
@test df[Rows, 2:3] == DataFrame(x=[2,3], y=["b", "c"])
@test df[Cols, 2] == ["a", "b", "c"]
@test df[Names] == [:x, :y]

A = [1 2; 3 4; 5 6];
@test A[Cols, 2] == [2; 4; 6]
@test A[Rows, 2:3] == [3 4; 5 6]

v = [1,2,3,4];
@test v[Rows, 2:3] == [2, 3]

@constant junk=KNNRegressor()

# we first test KNN as it will be used to check other functinality:
Xtr = [4 2 5 3;
     2 1 6 0.0];

@test MLJ.KNN.distances_and_indices_of_closest(3,
        MLJ.KNN.euclidean, Xtr, [1, 1])[2] == [2, 4, 1]

X = Xtr' |> collect
y = Float64[2, 1, 3, 8]
knn_ = KNNRegressor(K=3)
allrows = 1:4
estimator, state, report = fit(knn_, X, y, nothing, 0); # fit(model, X, y, state, verbosity)
@test report == nothing
@test estimator == state

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = [1.0 1.0]
ypred = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test isapprox(predict(knn_, estimator, Xtest)[1], ypred)


## TRAINABLE MODEL INTERFACE

X_frame, y = datanow(); # boston data
X = array(X_frame)

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y)
train, valid, test = partition(allrows, 0.7, 0.15)
@test vcat(train, valid, test) == allrows
knn = prefit(knn_, X, y)
fit!(knn, train);
the_error = rms(predict(knn, X[Rows, test]), y[test])

# TODO: compare to constant regressor and check it's significantly better


## DYNAMIC DATA INTERFACE

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn))

X, y = datanow(); # ALL of the data, training, test and validation
Xc = X
yc = y

# split the rows into training and testing rows:
allrows = eachindex(yc)
fold1, fold2 = partition(allrows, 0.7) # 70:30 split

Xd = dynamic(Xc)
yd = dynamic(yc)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
scale = prefit(scale_, Xd) # no need to fit

# get the transformed inputs, as if `scale` were already fit:
Xt = transform(scale, Xd)

# convert DataFrame Xt to an array:
Xa = array(Xt)

# choose a learner and make it trainable:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
knn = prefit(knn_, Xa, yd) # no need to fit

# get the predictions, as if `knn` already fit:
yhat = predict(knn, Xa)

# compute the error:
er = rms(yd, yhat)

# Now `er` is dynamic, so we can do "look through training" on any rows
# we like and evaluate on any rows:
fit!(er, fold1)
er[Rows, fold2]

fit!(er, fold2)
er_number = er[Rows, fold1]

# test echo:
Echo = MLJ.Echo
Xnew, ynew = datanow();
Xnew = Xnew[Rows, 3:7];
ynew = ynew[3:7];
@test X[Echo, Xnew] == Xnew
@test Xt[Echo, Xnew] == transform(scale, Xnew)
@test Xa[Echo, Xnew] == array(transform(scale, Xnew))
@test yhat[Echo, Xnew] == predict(knn, array(transform(scale, Xnew)))

include("Transformer.jl")
include("DecisionTree.jl")

