using Revise

using MLJ
using Test

import DataFrames: DataFrame, names

## UNIVERSAL DATA ADAPTOR

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

## KNN

# note: we first test KNN as it will be used to check other functionality

Xtr = [4 2 5 3;
     2 1 6 0.0];

@test MLJ.KNN.distances_and_indices_of_closest(3,
        MLJ.KNN.euclidean, Xtr, [1, 1])[2] == [2, 4, 1]

X = Xtr' |> collect
y = Float64[2, 1, 3, 8]
knn_ = KNNRegressor(K=3)
allrows = 1:4
fitresult, state, report = fit(knn_, X, y, nothing, 0); # fit(model, X, y, state, verbosity)
@test report == nothing
@test fitresult == state

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = [1.0 1.0]
ypred = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test isapprox(predict(knn_, fitresult, Xtest)[1], ypred)

include("dynamic.jl")
include("Transformer.jl")
include("DecisionTree.jl")

