# this file defines *and* loads one module

module KNN

export KNNRegressor

import MLJ: Regressor
using Dates
using LinearAlgebra

# to be extended:
import MLJ: predict, fit, clean!

KNNEstimatorType = Tuple{Matrix{Float64},Vector{Float64}, Dates.DateTime}

# TODO: introduce type parameters for the function fields (metric, kernel)

mutable struct KNNRegressor <: Regressor{KNNEstimatorType}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function # each target value is weighted by `kernel(distance^2)`
end

euclidean(v1, v2) = norm(v2 - v1)
reciprocal(d) = 1/d

# lazy keywork constructor:
function KNNRegressor(; K=1, metric=euclidean, kernel=reciprocal)
    model = KNNRegressor(K, metric, kernel)
    message = clean!(model)
    isempty(message) || @warn message
    return model
end
    
function clean!(model::KNNRegressor)
    message = ""
    if model.K <= 0
        model.K = 1
        message = message*"K cannot be negative. K set to 1."
    end
    return message
end

function fit(model::KNNRegressor
             , X::Matrix{Float64}
             , y::Vector{Float64}
             , state                     
             , verbosity) 
    
    # computing norms of rows later on is faster if we use the transpose of X:
    estimator = (X', y)
    state = estimator
    report = nothing
    
    return estimator, state, report 
end

first_component_is_less_than(v, w) = isless(v[1], w[1])

function distances_and_indices_of_closest(K, metric, Xtrain, pattern)

    distance_index_pairs = Array{Tuple{Float64,Int}}(undef, size(Xtrain, 2))
    for j in 1:size(Xtrain, 2)
        distance_index_pairs[j] = (metric(Xtrain[:,j], pattern), j)
    end

    sort!(distance_index_pairs, lt=first_component_is_less_than)
    distances = Array{Float64}(undef, K)
    indices = Array{Int}(undef, K)
    for j in 1:K
        distances[j] = distance_index_pairs[j][1]
        indices[j] = distance_index_pairs[j][2]
    end

    return distances, indices    
    
end

function predict_on_pattern(model, estimator, pattern)
    Xtrain, ytrain = estimator[1], estimator[2]
    distances, indices = distances_and_indices_of_closest(model.K, model.metric, Xtrain, pattern)
    wts = [model.kernel(d) for d in distances]
    wts = wts/sum(wts)
    return sum(wts .* ytrain[indices])
end

predict(model::KNNRegressor, estimator, Xnew) =
    [predict_on_pattern(model, estimator, Xnew[i,:]) for i in 1:size(Xnew,1)]
    
end # module


## EXPOSE THE INTERFACE

using .KNN



