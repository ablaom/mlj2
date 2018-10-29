task = load_iris();

# check some rudimentary task functions:

# get some binary classification data for testing
X, y = X_and_y(task)
X_array = convert(Array{Float64}, X);
 
import DecisionTree
baretree = DecisionTreeClassifier(target_type=String)

# split the rows:
allrows = eachindex(y)
train, test = partition(allrows, 0.7)
@test vcat(train, test) == allrows

estimator, state, report = fit(baretree, X_array, y, nothing, 1; display_tree=true);
@test report == nothing

# in this case decision tree is perfect predictor:
yhat = predict(baretree, estimator, X_array)
@test yhat == y

# test special features:
estimator, state, report = fit(baretree, X_array, y,
                               state, 0; prune_only=true, merge_purity_threshold=0.1)
yhat = predict(baretree, estimator, X_array)
@test yhat != y

