#using Revise
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

## PKG DecisionTree

import DecisionTree
baretree = DecisionTreeClassifier(target_type=String)

# split the rows:
all = eachindex(y)
train, test = partition(all, 0.7)
@test vcat(train, test) == all

estimator, state, report = fit(baretree, X_array, y, all, display_tree=true);
@test report == nothing

# in this case decision tree is perfect predictor:
yhat = predict(baretree, estimator, X_array)
@test yhat == y

# test special features:
estimator, state, report = fit(baretree, X_array, y, all,
                               state, prune_only=true, merge_purity_threshold=0.1)
yhat = predict(baretree, estimator, X_array)
@test yhat != y


## MODEL INTERFACE

@constant tree = prefit(baretree, X_array, y)
fit!(tree, all);
fit!(tree, all; prune_only=true, merge_purity_threshold=0.1, display_tree=true);
@test predict(tree, X_array) == yhat


## DYNAMIC DATA INTERFACE

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(tree))
matrixify(X) = convert(Array, X)
Xa = dynamic(matrixify, X)
@test isempty(tape(Xa))
yhat = dynamic(predict, tree, Xa)
@test tape(yhat) == MLJ.TrainableModel[tree]
Xa(all)
@test yhat(test) == y(test)





