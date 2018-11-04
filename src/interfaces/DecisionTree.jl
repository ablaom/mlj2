# this file defines *and* loads one module

#> This interface for the DecisionTree package is annotated so that it
#> may serve as a template for other supervised learning
#> interfaces. The annotations, which begin with "#>", should be
#> removed (but copy this file first!). See also the low-level
#> interface specification at "doc/adding_new_models.md".

#> Glue code goes in a module, whose name is the package name with
#> trailing underscore "_":
module DecisionTree_

#> export the new models you're going to define (and, if possible, nothing else):
export DecisionTreeClassifier

# to be extended:
import MLJ: predict, fit, fit2, clean!   #> compulsory for learners
# import MLJ: predict_proba              #> if implemented

# needed:
import DecisionTree                #> import package
import MLJ: Classifier, Regressor  #> and supertypes for the models to be defined

#> The DecisionTreeClassifier model type declared below is a
#> parameterized type (not necessary for models in general). This is
#> because the classifier built by DecisionTree.jl has a fit-result
#> type that depends on the target type, here denoted `T` (and the
#> fit-result type of a supervised model must be declared).

DecisionTreeClassifierFitResultType{T} =
    Union{DecisionTree.Node{Float64,T}, DecisionTree.Leaf{T}}

"""
[https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md](https://github.com/bensadeghi/DecisionTree.jl/blob/master/README.md)

"""
mutable struct DecisionTreeClassifier{T} <: Classifier{DecisionTreeClassifierFitResultType{T}} 
    pruning_purity::Float64 
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Float64
    display_depth::Int
end

# constructor:
#> all arguments are kwargs; 
#> give any compulsory args default value `nothing` and check they're
#> something
function DecisionTreeClassifier(
    ; target_type=nothing 
    , pruning_purity=1.0
    , max_depth=-1
    , min_samples_leaf=1
    , min_samples_split=2
    , min_purity_increase=0.0
    , n_subfeatures=0
    , display_depth=5)

    !(target_type == nothing) || throw(error("You must specify target_type=..."))

    model = DecisionTreeClassifier{target_type}(
        pruning_purity
        , max_depth
        , min_samples_leaf
        , min_samples_split
        , min_purity_increase
        , n_subfeatures
        , display_depth)

    message = clean!(model)           #> future proof by including these 
    isempty(message) || @warn message #> two lines even if no clean! defined below

    return model
end

#> The following optional method (the fallback does nothing, returns
#> empty warning) is called by the constructor above but also by the
#> fit methods below:
function clean!(model::DecisionTreeClassifier)
    warning = ""
    if  model.pruning_purity > 1
        warning *= "Need pruning_purity < 1. Resetting pruning_purity=1.0.\n"
        model.pruning_purity = 1.0
    end
    if model.min_samples_split < 2
        warning *= "Need min_samples_split < 2. Resetting min_samples_slit=2.\n"
        model.min_samples_split = 2
    end
    return warning
end

#> A required `fit` method returns `fitresult, cache, report`. The
#> `cache` is for use by `fit2` if implemented and `fit2` needs
#> it. Otherwise return `nothing` for `cache`.

function fit(model::DecisionTreeClassifier{T2}
             , verbosity            #> must be here even if unsupported in pkg (as here)
             , X::Array{Float64,2}
             , y::Vector{T}) where {T,T2}

    T == T2 || throw(ErrorException("Type, $T, of target incompatible "*
                                    "with type, $T2, of $model."))

    fitresult = DecisionTree.build_tree(y
                                        , X
                                        , model.n_subfeatures
                                        , model.max_depth
                                        , model.min_samples_leaf
                                        , model.min_samples_split
                                        , model.min_purity_increase)
    
    verbosity < 1 || DecisionTree.print_tree(fitresult, model.display_depth)

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be `nothing` or a dictionary keyed on symbols.
        
    cache = nothing
    report = nothing
    
    return fitresult, cache, report 

end

#> An optional `fit2` method, requiring the `fitresult` and `cache`
#> outputs of a previous `fit` on the same data (but on a possibly new
#> or mutated `model`) is for: (i) enabling refitting without
#> repeating some redundant computations; or (ii) requesting
#> package-specific functionality that depends on an initial train,
#> such as optimizing the weights in an ensemble, or, as here, pruning
#> a trained decision tree. There is a fall-back in MLJ which accepts
#> no keyword arguments and basically duplicates the effect of `fit`
#> above. See the "doc/adding_models.md" for more detail.

function fit2(model::DecisionTreeClassifier
              , verbosity           #> must be here even if unsupported in pkg
              , fitresult
              , cache
              , X                   #> no need to annotate types here
              , y  
              ; prune_only=false    #> kwargs to request post-train, pkg-specific actions
              , merge_purity_threshold=0.9)

    if prune_only

        verbosity < 1 || @info "Not completely retraining. Only pruning decision tree."
        
        fitresult = DecisionTree.prune_tree(fitresult, merge_purity_threshold)
        report = nothing
        cache = nothing
        
        verbosity < 1 || DecisionTree.print_tree(fitresult, model.display_depth)

#        verbosity < 1 || @info "Done."

        return fitresult, cache, report

    else # fit as usual:

        return fit(model, verbosity, X, y)

    end

end

predict(model::DecisionTreeClassifier 
        , fitresult
        , Xnew::Union{Array{Float64,2},SubArray{Float64,2}}) =
            DecisionTree.apply_tree(fitresult, collect(Xnew))


end # module


## EXPOSE THE INTERFACE

using .DecisionTree_
export DecisionTreeClassifier         

