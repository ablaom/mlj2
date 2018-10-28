using Revise
using Test
using MLJ

Xt = [4 2 5 3;
     2 1 6 0.0]

# we first test KNN as it will be used to check other functinality:
@test MLJ.KNN.distances_and_indices_of_closest(3,
        MLJ.KNN.euclidean, Xt, [1, 1])[2] == [2, 4, 1]

X = Xt'
y = Float64[2, 1, 3, 8]
knn_ = KNNRegressor(K=3)
allrows = 1:4
estimator, state, report = fit(knn_, X, y, allrows, nothing, 0); # fit(model, X, y, rows, state, verbosity)
@test report == nothing
@test estimator == state

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = [1.0 1.0]
yhat = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test isapprox(predict(knn_, estimator, Xtest)[1], yhat)


## MODEL INTERFACE

array(X) = convert(Array, X)

X_frame, y = datanow(); # boston data
X = array(X_frame);

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y)
train, valid, test = partition(allrows, 0.7, 0.15)
@test vcat(train, valid, test) == allrows
@constant knn = prefit(knn_, X, y)
fit!(knn, train);
er = rms(predict(knn, X(test)), y(test))

# TODO: compare to constant regressor and check it's significantly better


## DYNAMIC DATA INTERFACE

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn))
Xa = dynamic(array, X)
@test isempty(tape(Xa))


yhat = dynamic(predict, knn, Xa)
@test tape(yhat) == MLJ.TrainableModel[knn]
Xa(allrows)
@test yhat(test) == y(test)

# can do this:
fit!(yhat, train, display_tree=true);
pred1 = yhat(test)

# or this:
fit!(knn, train; display_tree=true);
pred2 = predict(knn, X_array(test))

# shouldn't make a difference:
@test pred1 == pred2

include("Transformer.jl")
include("DecisionTree.jl")


