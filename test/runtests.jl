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

include("KNN.jl")
include("dynamic.jl")
include("Transformer.jl")
include("DecisionTree.jl")

