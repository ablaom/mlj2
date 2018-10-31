# this file defines *and* loads one module

#> This interface for the DecisionTree package is annotated so that it
#> may serve as a template for other supervised learning
#> interfaces. The annotations, which begin with "#>", should be
#> removed (but copy this file first!)

#> TODO: checklist, incl, "registering" interface with MLJ

#> presently module name should be the package name with trailing underscore "_". 
module DecisionTree_

#> export the new models you're going to define:
export DecisionTreeClassifier

# to be extended:
import MLJ: predict, fit, clean!   #> compulsory
# import MLJ: predict_proba        #> if implemented

# needed:
import DecisionTree                #> import package
import MLJ: Classifier, Regressor  #> and supertypes for the models to be defined


#> The following bare model type is a parameterized type (not true in
#> general). This is because: (1) the classifier built by
#> DecisionTree.jl has a estimator type that depends on the the target
#> type, here denoted `T`; and (2) our universal requirement that the
#> types of estimators be declared (by specifying the parameter of the
#> bare model supertype, in this case `Classifier`).

# estimator could be a stump, which has a different type from regular
# tree; therefore, estimators are instances of a union type:
DecisionTreeClassifierEstimatorType{T} =
    Union{DecisionTree.Node{Float64,T}, DecisionTree.Leaf{T}}
mutable struct DecisionTreeClassifier{T} <: Classifier{DecisionTreeClassifierEstimatorType{T}} 
    pruning_purity::Float64 
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Float64
    frozen::Bool   #> every model must have this field; for
                     #> suppressing training in learning networks
end

# constructor:
#> all arguments are kwargs; 
#> give compulsory args default value `nothing` and check they're something
function DecisionTreeClassifier(
    ; target_type=nothing 
    , pruning_purity=1.0
    , max_depth=-1
    , min_samples_leaf=1
    , min_samples_split=2
    , min_purity_increase=0.0
    , n_subfeatures=0
    , frozen=false)

    !(target_type == nothing) || throw(error("You must specify target_type=..."))

    learner = DecisionTreeClassifier{target_type}(
        pruning_purity
        , max_depth
        , min_samples_leaf
        , min_samples_split
        , min_purity_increase
        , n_subfeatures
        , frozen)

    message = clean!(learner)         #> future proof by including these 
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return learner
end

#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods exposed by MLJ:
function clean!(learner::DecisionTreeClassifier)
    warning = ""
    if  learner.pruning_purity > 1
        warning *= "Need pruning_purity < 1. Resetting pruning_purity=1.0.\n"
        learner.pruning_purity = 1.0
    end
    if learner.min_samples_split < 2
        warning *= "Need min_samples_split < 2. Resetting min_samples_slit=2.\n"
        learner.min_samples_split = 2
    end
    return warning
end

#> a `fit` method returns (estimator, state, report)
function fit(learner::DecisionTreeClassifier{T2}
             , X::Array{Float64,2}
             , y::Vector{T}
             , state                     #> pkg-specific features require state
             , verbosity         #> must be here even if unsupported in pkg (as here)
             ; display_tree=true #> kwargs accesss pkg-specific features
             , display_depth=5
             , prune_only=false
             , merge_purity_threshold=0.9) where {T, T2}

    T == T2 || throw(ErrorException("Type, $T, of target incompatible "*
                                    "with type, $T2, of $learner."))

    if prune_only
        state != nothing || throw(error("Cannot prune without state."))
        estimator = DecisionTree.prune_tree(state, merge_purity_threshold)

        #> return output of package-specific functionality (eg,
        #> feature rankings, internal estimates of generalization error)
        #> in `report`, which should be `nothing` or a dictionary
        #> keyed on symbols:
        report = Dict{Symbol,Any}()
        report[:last_prune] = "Tree last pruned with merge_purity_threshold=$merge_purity_threshold."

        state = estimator

        return estimator, state, report
    end

    #> would have passed verbosity level below had it been supported.
    #> supported.
    estimator = DecisionTree.build_tree(
        y
        , X
        , learner.n_subfeatures
        , learner.max_depth
        , learner.min_samples_leaf
        , learner.min_samples_split
        , learner.min_purity_increase)

    !display_tree || DecisionTree.print_tree(estimator, display_depth)

    state = estimator

    report = nothing
    
    return estimator, state, report 
end

predict(learner::DecisionTreeClassifier 
        , estimator
        , Xnew::Union{Array{Float64,2},SubArray{Float64,2}}) = DecisionTree.apply_tree(estimator, collect(Xnew))


end # module


## EXPOSE THE INTERFACE

using .DecisionTree_
export DecisionTreeClassifier         

