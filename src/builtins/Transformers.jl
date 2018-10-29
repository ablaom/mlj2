# note: this file defines *and* imports one module; see end
module Transformers

export ToIntTransformer
export UnivariateStandardizer, Standardizer

import MLJ: MLJType, Transformer
import DataFrames: names, AbstractDataFrame, DataFrame, eltypes
import Distributions
using Statistics

# to be extended:
import MLJ: fit, transform, inverse_transform


## CONSTANTS

const N_VALUES_THRESH = 16 # for BoxCoxTransformation


## FOR RELABELLING BY CONSECUTIVE INTEGERS STARTING AT 1

mutable struct ToIntTransformer <: Transformer
    sorted::Bool
    initial_label::Int # ususally 0 or 1
    map_unseen_to_minus_one::Bool # unseen inputs are transformed to -1
end

ToIntTransformer(; sorted=true, initial_label=1,
                 map_unseen_to_minus_one=false) =
                     ToIntTransformer(sorted, initial_label,
                                      map_unseen_to_minus_one)

struct ToIntEstimator{T} <: MLJType
    n_levels::Int
    int_given_T::Dict{T, Int}
    T_given_int::Dict{Int, T}
end

# null estimator constructor:
ToIntEstimator(S::Type{T}) where T =
    ToIntEstimator{T}(0, Dict{T, Int}(), Dict{Int, T}())

function fit(transformer::ToIntTransformer
             , v::AbstractVector{T}
             , rows::AbstractVector{Int}
             , state
             , verbosity) where T

    int_given_T = Dict{T, Int}()
    T_given_int = Dict{Int, T}()
    vals = collect(Set(v[rows])) 
    if transformer.sorted
        sort!(vals)
    end
    n_levels = length(vals)
    if n_levels > 2^62 - 1
        error("Cannot encode with integers a vector "*
                         "having more than $(2^62 - 1) values.")
    end
    i = transformer.initial_label
    for c in vals
        int_given_T[c] = i
        T_given_int[i] = c
        i = i + 1
    end

    estimator = ToIntEstimator{T}(n_levels, int_given_T, T_given_int)
    state = estimator
    report = nothing

    return estimator, state, report

end

# scalar case:
function transform(transformer::ToIntTransformer, estimator::ToIntEstimator{T}, x::T) where T
    ret = 0 # otherwise ret below stays in local scope
    try 
        ret = estimator.int_given_T[x]
    catch exception
        if isa(exception, KeyError)
            if transformer.map_unseen_to_minus_one 
                ret = -1
            else
                throw(exception)
            end
        end 
    end
    return ret
end 
inverse_transform(transformer::ToIntTransformer, estimator, y::Int) =
    estimator.T_given_int[y]

# vector case:
function transform(transformer::ToIntTransformer, estimator::ToIntEstimator{T},
                   v::AbstractVector{T}) where T
    return Int[transform(transformer, estimator, x) for x in v]
end
inverse_transform(transformer::ToIntTransformer, estimator::ToIntEstimator{T},
                  w::AbstractVector{Int}) where T = T[estimator.T_given_int[y] for y in w]


## UNIVARIATE STANDARDIZATION

struct UnivariateStandardizer <: Transformer end

function fit(transformer::UnivariateStandardizer, v::AbstractVector{T}, rows,
             state, verbosity) where T <: Real
    std(v) > eps(Float64) || 
        @warn "Extremely small standard deviation encountered in standardization."
    estimator = (mean(v[rows]), std(v[rows]))
    state = estimator
    report = nothing
    return estimator, state, report
end

# for transforming single value:
function transform(transformer::UnivariateStandardizer, estimator, x::Real)
    mu, sigma = estimator
    return (x - mu)/sigma
end

# for transforming vector:
transform(transformer::UnivariateStandardizer, estimator,
          v::AbstractVector{T}) where T <: Real =
              [transform(transformer, estimator, x) for x in v]

# for single values:
function inverse_transform(transformer::UnivariateStandardizer, estimator, y::Real)
    mu, sigma = estimator
    return mu + y*sigma
end

# for vectors:
inverse_transform(transformer::UnivariateStandardizer, estimator,
                  w::AbstractVector{T}) where T <: Real =
                      [inverse_transform(transformer, estimator, y) for y in w]


## STANDARDIZATION OF ORDINAL FEATURES OF A DATAFRAME

# TODO: reimplement in simpler, safer way: estimator is two vectors:
# one of features that are transformed, one of corresponding
# univariate trainable models.

""" Standardizers the columns of eltype <: AbstractFloat unless non-empty `features` specfied."""
mutable struct Standardizer <: Transformer
    features::Vector{Symbol} # features to be standardized; empty means all of 
end

# lazy keyword constructor:
Standardizer(; features=Symbol[]) = Standardizer(features)

struct StandardizerEstimator <: MLJType
    estimators::Matrix{Float64}
    features::Vector{Symbol} # all the feature labels of the data frame fitted
    is_transformed::Vector{Bool}
end

# null estimator:
StandardizerEstimator() = StandardizerEstimator(zeros(0,0), Symbol[], Bool[])

function fit(transformer::Standardizer, X::AbstractDataFrame, rows, state, verbosity)

    Xv = view(X, rows)
    features = names(Xv)
    
    # determine indices of features to be transformed
    features_to_try = (isempty(transformer.features) ? features : transformer.features)
    is_transformed = Array{Bool}(undef, size(Xv, 2))
    for j in 1:size(Xv, 2)
        if features[j] in features_to_try && eltype(Xv[j]) <: AbstractFloat
            is_transformed[j] = true
        else
            is_transformed[j] = false
        end
    end

    # fit each of those features
    estimators = Array{Float64}(undef, 2, size(Xv, 2))
    verbosity < 1 || @info "Features standarized: "
    n_rows = size(Xv, 1)
    for j in 1:size(Xv, 2)
        if is_transformed[j]
            estimator, state, report =
                fit(UnivariateStandardizer(), collect(Xv[j]), 1:n_rows, nothing, verbosity - 1)
            estimators[:,j] = [estimator...]
            verbosity < 1 ||
                @info "  :$(features[j])    mu=$(estimators[1,j])  sigma=$(estimators[2,j])"
        else
            estimators[:,j] = Float64[0.0, 0.0]
        end
    end
    
    estimator = StandardizerEstimator(estimators, features, is_transformed)
    state = estimator
    report = Dict{Symbol,Any}()
    report[:features_transformed]=[features[is_transformed]]
    
    return estimator, state, report
    
end

function transform(transformer::Standardizer, estimator, X)

    names(X) == estimator.features ||
        error("Attempting to transform data frame with incompatible feature labels.")

    Xnew = X[1:end,:] # make a copy of X, working even for `SubDataFrames`
    univ_transformer = UnivariateStandardizer()
    for j in 1:size(X, 2)
        if estimator.is_transformed[j]
            # extract the (mu, sigma) pair:
            univ_estimator = (estimator.estimators[1,j], estimator.estimators[2,j])  
            Xnew[j] = transform(univ_transformer, univ_estimator, collect(X[j]))
        end
    end
    return Xnew

end    

end # end module


## EXPOSE THE INTERFACE

using .Transformers


