# this file defines *and* loads one module

module KNN

export KNNRegressor

import MLJ: Regressor
using Dates
using LinearAlgebra

# to be extended:
import MLJ: predict, fit, fit2, clean!

KNNFitResultType = Tuple{Matrix{Float64},Vector{Float64}, Dates.DateTime}

# TODO: introduce type parameters for the function fields (metric, kernel)

mutable struct KNNRegressor <: Regressor{KNNFitResultType}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function 
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
             , verbosity
             , X::Matrix{Float64}
             , y::Vector{Float64})
    
    # computing norms of rows later on is faster if we use the transpose of X:
    fitresult = (X', y)
    cache = nothing
    report = nothing
    
    return fitresult, cache, report 
end

fit2(model, verbosity, fitresult, cache, X, y) = fitresult, cache, nothing

first_component_is_less_than(v, w) = isless(v[1], w[1])

# TODO: there is way smarter way to do without sorting. Alternatively,
# get a KNN solver from external package.
function distances_and_indices_of_closest(K, metric, Xtrain, pattern)

    distance_index_pairs = Array{Tuple{Float64,Int}}(undef, size(Xtrain, 2))
    for j in 1:size(Xtrain, 2)
        distance_index_pairs[j] = (metric(Xtrain[:,j], pattern), j)
    end

    sort!(distance_index_pairs, lt=first_component_is_less_than)
    Kprime = min(K,size(Xtrain, 2)) # in case less patterns than K
    distances = Array{Float64}(undef, Kprime)
    indices = Array{Int}(undef, Kprime)
    for j in 1:Kprime
        distances[j] = distance_index_pairs[j][1]
        indices[j] = distance_index_pairs[j][2]
    end

    return distances, indices    
    
end

function predict_on_pattern(model, fitresult, pattern)
    Xtrain, ytrain = fitresult[1], fitresult[2]
    distances, indices = distances_and_indices_of_closest(model.K, model.metric, Xtrain, pattern)
    wts = [model.kernel(d) for d in distances]
    wts = wts/sum(wts)
    return sum(wts .* ytrain[indices])
end

predict(model::KNNRegressor, fitresult, Xnew) =
    [predict_on_pattern(model, fitresult, Xnew[i,:]) for i in 1:size(Xnew,1)]
    
end # module


## EXPOSE THE INTERFACE

using .KNN



