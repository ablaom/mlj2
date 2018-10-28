using Revise
using Test
using MLJ

# check methods in datanow.jl and RegressorTask, ClassifierTask
# constructors:
# load_boston();
# load_ames();
task = load_iris();

# check some rudimentary task functions:

# get some binary classification data for testing
X, y = X_and_y(task)
X_array = convert(Array{Float64}, X);

## PKG INTERFACE: DecisionTree

import DecisionTree
baretree = DecisionTreeClassifier(target_type=String)

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7)
@test vcat(train, test) == allrows

estimator, state, report = fit(baretree, X_array, y, allrows, nothing, true, display_tree=true);
@test report == nothing

# in this case decision tree is perfect predictor:
yhat = predict(baretree, estimator, X_array)
@test yhat == y

# test special features:
estimator, state, report = fit(baretree, X_array, y, allrows,
                               state, false, prune_only=true, merge_purity_threshold=0.1)
yhat = predict(baretree, estimator, X_array)
@test yhat != y


## MODEL INTERFACE

@constant tree = prefit(baretree, X_array, y)
fit!(tree, allrows);
fit!(tree, allrows; prune_only=true, merge_purity_threshold=0.1, display_tree=true);
@test predict(tree, X_array) == yhat


## DYNAMIC DATA INTERFACE

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(tree))
array(X) = convert(Array, X)
Xa = dynamic(array, X)
@test isempty(tape(Xa))
yhat = dynamic(predict, tree, Xa)
@test tape(yhat) == MLJ.TrainableModel[tree]
Xa(allrows)
@test yhat(test) == y(test)

# can do this:
fit!(yhat, train, display_tree=true);
pred1 = yhat(test)

# or this:
fit!(tree, train; display_tree=true);
pred2 = predict(tree, X_array(test))

# shouldn't make a difference: BUT IT DOES; DECISION TREE IS NOT DETERMINISTIC!!!
#@test pred1 == pred2


## BUILTIN: transformers.jl

# relabelling with integer transformer:
to_int_hypers = ToIntTransformer()
to_int = prefit(to_int_hypers, y)
fit!(to_int, allrows)
z = transform(to_int, y[test])
@test y[test] == inverse_transform(to_int, z)
to_int_hypers.map_unseen_to_minus_one = true
to_int = prefit(to_int_hypers, [1,2,3,4])
fit!(to_int, [1,2,3,4])
@test transform(to_int, 5) == -1
@test transform(to_int, [5,1])[1] == -1 

# `UnivariateStandardizer`:
stand = prefit(UnivariateStandardizer(), [0, 2, 4])
fit!(stand, 1:3)
@test round.(Int, transform(stand, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, [-1, 1, 3])) == [0, 4, 8] 
X, y = X_and_y(load_ames());
train, test = partition(eachindex(y), 0.9);

# introduce a field of type `Char`:
X[:OverallQual] = map(Char, X[:OverallQual]);

# `Standardizer`:
stand = prefit(Standardizer(), X)
fit!(stand, eachindex(y))
transform(stand, X)

stand = prefit(Standardizer(features=[:GrLivArea]), X)
fit!(stand, eachindex(y))
transform(stand, X)


