module MLJ

export glossary
export @more, @constant
export predict, fit, transform, inverse_transform, fit!
export features, X_and_y
export load_boston, load_ames, load_iris
export TrainableModel, prefit, dynamic
export partition

import Requires.@require  # lazy code loading package
import CSV
import DataFrames: names, DataFrame, AbstractDataFrame, SubDataFrame


# CONSTANTS

const COLUMN_WIDTH = 24           # for displaying dictionaries with `show`
const DEFAULT_SHOW_DEPTH = 2      # how deep to display fields of `MLJType` objects
const srcdir = dirname(@__FILE__) # the directory containing this file 


## GLOSSARY

# A bare-bones definition of terms for the Julia-illiterate, light on
# implementation details:

"""
### task (object of type `Task`)

Data plus a clearly specified learning objective. In addition, a
description of how the completed task is to be evaluated.


### hyperparameters

Parameters on which some learning algorithm depends, specified before
the algorithm is applied, and where learning is interpreted in the
broadest sense. For example, PCA feature reduction is a
"preprocessing" transformation "learning" a projection from training
data, governed by a dimension hyperparameter. Hyperparameters in our
sense may specify configuration (eg, number of parallel processes)
even when this does not effect the end-product of learning. (We do
exclude logging configuration parameters, eg verbosity level.)


### model (object of abstract type `Model`)

Object collecting together hyperameters of a single algorithm. 


### learner (object of abstract type `Learner`)

Informally, any learning algorithm. More technically, a model
associated with such an algorithm.


### transformer (object of abstract type `Transformer`)

Informally, anything that transforms data or an algorithm that
"learns" such transforms from training data (eg, feature reduction,
normalization). Or, more technically, the model associated with
such an algorithm.


### estimator 

The "weights" or "parmaters" learned by an algorithm using the
hyperparameters prescribed in an associated model (eg, what a learner
needs to predict or what a transformer needs to transform). An
estimator may have a Julia type determined by a package external to
MLJ, but which will be declared in the MLJ interface for that package
(important for efficient implementation of ensemble learners).


### method

What Julia calls a function. (In Julia, a "function" is a collection
of methods sharing the same name but different type signatures.)


### operator

Data-manipulating operations (methods) parameterized by some
estimator. For learners, the `predict` or `predict_proba` methods, for
transformers, the `transform` or `inverse_transform` method. In some
contexts such an operator might be replaced by an ordinary operator
(methods) that do *not* depend on an estimator, which are then then
called *static* operators for clarity. An operator that is not static
is *dynamic*.


### state (type not presently constrained)

A product of training that is sufficient to perform any implemented
learner/transformer-specific functions beyond normal training,
transforming, or predicting (eg, pruning an existing decision tree,
optimization weights of an ensemble learner). In particular, in the
case of iterative learners, state must be sufficient to restart the
training algorithm (eg, add decision trees to a random forest). The
estimator may serve as state and should do so by default.


### trainable model

An object consisting of a model wrapped with an estimator, state,
*and* a source for training data. A *source* is either concrete data
(eg, data frame) or *dynamic data*, as defined below. The estimator
and state are undefined until the model is trained.

A trainable model might also include metadata recording algorithm-specific
statistics of training (eg, internal estimate of generalization error)
or the results of calling the `fit` method with special instructions
(eg, `calculate_feature_importance=true`).


### dynamic data

A "trainable" data-like object consisting of:

(1) An **operator**, static or dynamic.

(2) A **trainable model**, void if the operator is static.

(3) Connections to other dynamic or static data, specified by a list
   of **arguments** (one for each argument of the operator); each
   argument is data, dynamic or static.

(4) An **activity flag** used to switch dynamic behaviour on or off.

(5) Metadata tracking the object's dependency on various estimators,
    as implied by its connections.


### learning network (implicity defined by dynamic data)

A directed graph implicit in the specification of dynamic data. All
nodes are dynamic data except for the source nodes, which are
static. Something like a scikit-learn pipeline.

"""
glossary() = nothing


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# overload `show` method for MLJType (which becomes the fallback for
# all subtypes) and define REPL macro `@more` to extract detail, and
# `@constant` macro for assigning to new variables in way that
# "registers" the binding:
include("show.jl")

# for storing hyperparameters associated with a estimator of type `E` (see
# above for definition of estimator):
abstract type Model{E} <: MLJType end 

# a model type for learners:
abstract type Learner{E} <: Model{E} end

# a model type for transformers
abstract type Transformer{E} <: Model{E} end 

# special learners:
abstract type Supervised{E} <: Learner{E} end
abstract type Unsupervised{E} <: Learner{E} end

# special supervised learners:
abstract type Regressor{E} <: Supervised{E} end
abstract type Classifier{E} <: Supervised{E} end

# tasks:
abstract type Task <: MLJType end 


## METHODS TO CLOSE GAP BETWEEN `Array`s, `DataFrame`s AND JuliaDB `Table`s

# note: for now trying to avoid complicated data interface to accommodate
# a variety of data containers but ultimately this may be unavoidable

select(df::AbstractDataFrame, selection) = df[selection]
(df::DataFrame)(rows) = view(df, rows)
(df::SubDataFrame)(rows) = view(df, rows)
(v::Vector)(rows) = v[rows]
#  names(t::Table) = isempty(table) ? Symbol[] : getfields(t[1])
# (t::Table)(rows) = t[rows]


## CONCRETE TASK TYPES

mutable struct SupervisedTask <: Task
    kind::Symbol
    data
    target::Symbol
    ignore::Vector{Symbol}
end

function SupervisedTask(
    ; kind=nothing
    , data=nothing
    , target=nothing
    , ignore=Symbol[])

    kind != nothing         || @error "You must specfiy kind=..."
    data != nothing         || @error "You must specify data=..."
    target != nothing       || @error "You must specify target=..."
    target in names(data)   || @error "Supplied data does not have $target as field."
    return SupervisedTask(kind, data, target, ignore)
end

ClassificationTask(; kwargs...) = SupervisedTask(; kind=:classification, kwargs...)
RegressionTask(; kwargs...)     = SupervisedTask(; kind=:regression, kwargs...)


## RUDIMENTARY TASK OPERATIONS

features(task::Task) = filter!(names(task.data)) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(names(task.data)) do ftr
    ftr != task.target && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = select(task.data, features(task)), select(task.data, task.target)


## SOME LOCALLY ARCHIVED TASKS FOR TESTING AND DEMONSTRATION

# define `load_ames()`, `load_boston()` and `load_iris()`:
include("datanow.jl")


## PACKAGE INTERFACE METHODS 

# to be extended by interfaces:
function fit end
function predict end
function predict_proba end
function transform end 
function inverse_transform end

# fallback method to correct invalid hyperparameters and return
# a warning (in this case empty):
clean!(estimator::Model) = "" 

# note: package interfaces define concrete `Model` types, the
# users' "basement level" abstractions.

# note: package interfaces are contained in "/src/interfaces.jl" and
# lazy loaded. built-in model definitions and associated methods (ie,
# ones not dependent on external packages) are contained in
# "/src/builtins.jl"


## MODEL INTERFACE - BARE BONES

""" 
    merge!(v1::Vector, v2::Vector)

Append to `v1` all elements of `v2` not in `v1`, in the
order they appear, and return result.

"""
function merge!(v1::Vector, v2::Vector)
    for x in v2
        if !(x in v1)
            push!(v1, x)
        end
    end
    return v1
end

# TODO: replace linear tapes below with dependency trees to allow
# better scheduling of training dynamic data.

mutable struct TrainableModel{B<:Model} <: MLJType

    model::B
    estimator
    state
    args
    report
    tape::Vector{TrainableModel}
    
    function TrainableModel{B}(model::B, args...) where B<:Model

        trainable = new{B}(model)
        trainable.args = args

        # note: `get_tape(arg)` returns arg.tape where this makes
        # sense and an empty tape otherwise.  However, the complete
        # definition of `get_tape` must be postponed until
        # `DynamicData` type is defined.

        # combine the tapes of all arguments to make a new tape:
        tape = get_tape(nothing) # returns blank tape 
        for arg in args
            merge!(tape, get_tape(arg))
        end
        trainable.tape = tape

        return trainable
    end
end

# automatically detect type parameter:
TrainableModel(model::B, args...) where B<:Model = TrainableModel{B}(model, args...)

# fit method:
function fit!(trainable::TrainableModel, rows; verbosity=1, kwargs...)
    verbosity < 1 || @info "Training $trainable."
    if isdefined(trainable, :state)
        trainable.estimator, trainable.state, trainable.report =
            fit(trainable.model, trainable.args..., rows, trainable.state; verbosity=verbosity-1, kwargs...)
    else
        trainable.estimator, trainable.state, trainable.report =
            fit(trainable.model, trainable.args..., rows; verbosity=verbosity-1, kwargs...)
    end
    verbosity <1 || @info "Done."
end


## MODEL INTERFACE - SPECIFIC TO SUPERVISED LEARNERS

# trainable model constructor for supervised models:
function prefit(model::S, X=nothing, y=nothing) where S<:Supervised
    !(X == nothing || y == nothing) ||
        @warn "To make trainable instead use `prefit(model, X, y)` or later call `model.args = X, y`."
    return TrainableModel(model, X, y)
end

# predict method for learner models:
function predict(model::TrainableModel{L}, X) where L<:Learner
    if isdefined(model, :estimator)
        return predict(model.model, model.estimator, X)
    else
        throw(error("$model is not trained and so cannot predict."))
    end
end

# TODO: predict_proba method for classifier models:


## MODEL INTERFACE - SPECIFIC TO UNSUPERVISED LEARNERS

# TODO


## MODEL INTERFACE - SPECIFIC TO TRANSFORMERS

# trainable model constructor for transformers:
function prefit(model::T, X=nothing) where T<:Transformer
    X != nothing ||
        @warn "To make trainable instead use `prefit(model, X)` or later call `model.args = X`."
    return TrainableModel(model, X)
end

function transform(model::TrainableModel{L}, X) where L<:Learner
    if isdefined(model, :estimator)
        return transform(model.model, X)
    else
        throw(error("$model is not trained and so cannot transform."))
    end
end

function inverse_transform(model::TrainableModel{L}, X) where L<:Learner
    if isdefined(model, :estimator)
        return inverse_transform(model.model, X)
    else
        throw(error("$model is not trained and so cannot inverse_transform."))
    end
end


## DYNAMIC DATA INTERFACE - BASICS

struct DynamicData{M<:Union{TrainableModel, Nothing}} <: MLJType

    operator::Function              # eg, `predict` or `inverse_transform` or a static operator
    trainable::M                        # is `nothing` for static operators
    args                            # data (static or dynamic) furnishing inputs for `operator`
    tape::Vector{TrainableModel}   # for tracking dependencies

    function DynamicData{M}(operator, trainable::M, args...) where M<:Union{TrainableModel, Nothing}

        # get the trainable model's dependencies:
        tape = copy(get_tape(trainable))

        # add the trainable model itself as a dependency:
        if trainable != nothing
            merge!(tape, [trainable, ])
        end
        
        # append the dependency tapes of all arguments:
        for arg in args
            merge!(tape, get_tape(arg))
        end

        return new{M}(operator, trainable, args, tape)

    end
end

# to complete the definition of `TrainableModel` and `DynamicData`
# constructors:
get_tape(::Any) = TrainableModel[]
get_tape(X::DynamicData) = X.tape
get_tape(trainable::TrainableModel) = trainable.tape

# autodetect type parameter:
DynamicData(operator, trainable::M, args...) where M<:Union{TrainableModel, Nothing} =
    DynamicData{M}(operator, trainable, args...)

# to make dispatch unambiguous in "dynamic" constructors appearing later:

# user's fall-back constructor:
dynamic(operator::Function, trainable::TrainableModel, args...) = DynamicData(operator, trainable, args...)

# user's constructor for learners and transformers:
LT = Union{Learner,Transformer}
function dynamic(operator::Function, trainable::TrainableModel{M}, args...) where M<:LT
    length(args) == 1 || throw(error("Wrong number of arguments. "*
                                     "Use `dynamic(operator, trainable_model, X)` for learners or transfomers."))
    return DynamicData(operator, trainable, args...)
end

# user's special constructor for static operators:
function dynamic(operator::Function, args...)
    length(args) > 0 || throw(error("`args` in `dynamic(::Function, args...)` must be non-empty. "))
    trainable = nothing
    return DynamicData(operator, trainable, args...)
end

# make dynamic data callable:
(X::DynamicData)(rows) =
    (X.operator)(X.trainable, [X.args[j](rows) for j in eachindex(X.args)]...)

# special case of static operators:
(X::DynamicData{Nothing})(rows) =
    (X.operator)([X.args[j](rows) for j in eachindex(X.args)]...)


## MISCELLANEOUS TOOLS

"""
    partition(rows::AbstractVector{Int}, fractions...)

Splits the vector `rows` into a tuple of vectors whose lengths are
given by the corresponding `fractions` of `length(rows)`. The last
fraction is not provided, as it is inferred from the preceding
ones. So, for example,

    julia> partition(1:1000, 0.2, 0.7)
    (1:200, 201:900, 901:1000)

"""
function partition(rows::AbstractVector{Int}, fractions...)
    rows = collect(rows)
    rowss = []
    if sum(fractions) >= 1
        throw(DomainError)
    end
    n_patterns = length(rows)
    first = 1
    for p in fractions
        n = round(Int, p*n_patterns)
        n == 0 ? (@warn "A split has only one element"; n = 1) : nothing
        push!(rowss, rows[first:(first + n - 1)])
        first = first + n
    end
    if first > n_patterns
        @warn "Last vector in the split has only one element."
        first = n_patterns
    end
    push!(rowss, rows[first:n_patterns])
    return tuple(rowss...)
end


## SETUP LAZY PKG INTERFACE LOADING

# note: presently an MLJ interface to a package, eg `DecisionTree`,
# is not loaded by `using MLJ` alone; one must additionally call
# `import DecisionTree`. 

# files containing a pkg interface must have same name as pkg plus ".jl"

macro load_interface(pkgname, uuid::String, load_instr)
    (load_instr.head == :(=) && load_instr.args[1] == :lazy) || throw(error("Invalid load instruction"))
    lazy = load_instr.args[2]
    filename = joinpath("interfaces", string(pkgname, ".jl"))

    if lazy
        quote
            @require $pkgname=$uuid include($filename)
        end
    else
        quote
            @eval include(joinpath($srcdir, $filename))
        end
    end
end

function __init__()
    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true
end

end # module



    
