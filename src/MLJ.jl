module MLJ

export glossary
export features, X_and_y
export rms, rmsl, rmslp1, rmsp
export load_boston, load_ames, load_iris, datanow
export TrainableModel, prefit, dynamic, fit!
export array

# defined here but extended by files in "interfaces/" (lazily loaded)
export predict, fit, transform, inverse_transform

# defined in include file "utilities.jl":
export partition, @curve, @pcurve # and @colon

# defined in include file "show.jl":
export @more, @constant

# defined in include file "builtins/Transformers.jl":
export ToIntTransformer
export UnivariateStandardizer, Standardizer
# export OneHotEncoder
# export UnivariateBoxCoxTransformer, BoxCoxTransformer
# export DataFrameToArrayTransformer, RegressionTargetTransformer
# export MakeCategoricalsIntTransformer
# export DataFrameToStandardizedArrayTransformer
# export IntegerToInt64Transformer
# export UnivariateDiscretizer, Discretizer

# defined in include file "builtins/KNN.jl":
export KNNRegressor

import Requires.@require  # lazy code loading package
import CSV
import DataFrames: DataFrame, AbstractDataFrame, SubDataFrame, eltypes, names
import Distributions

# from Standard Library:
using Statistics
using LinearAlgebra


# CONSTANTS

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
needs to predict or what a transformer needs to transform). 


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

include("utilities.jl")



## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# overload `show` method for MLJType (which becomes the fall-back for
# all subtypes):
include("show.jl")

# for storing hyperparameters:
abstract type Model <: MLJType end 

# a subtype for learners with an associated estimator of type
# `E`; type parameter needed to make ensemble learners efficient:
abstract type Learner{E} <: Model end

# a model type for transformers
abstract type Transformer <: Model end 

# special learners:
abstract type Supervised{E} <: Learner{E} end
abstract type Unsupervised{E} <: Learner{E} end

# special supervised learners:
abstract type Regressor{E} <: Supervised{E} end
abstract type Classifier{E} <: Supervised{E} end

# tasks:
abstract type Task <: MLJType end 


## LOSS AND LOW-LEVEL ERROR FUNCTIONS

function rms(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(rows))
end

function rms(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end

function rmsl(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(rows))
end

function rmsl(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end

function rmslp1(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(rows))
end

function rmslp1(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end

""" Root mean squared percentage loss """
function rmsp(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    count = 0
    for i in rows
        if y[i] != 0.0
            dev = (y[i] - yhat[i])/y[i]
            ret += dev*dev
            count += 1
        end
    end
    return sqrt(ret/count)
end

function rmsp(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    count = 0
    for i in eachindex(y)
        if y[i] != 0.0
            dev = (y[i] - yhat[i])/y[i]
            ret += dev*dev
            count += 1
        end
    end
    return sqrt(ret/count)
end

# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, yhat::AbstractVector{T}) where T<:Real = 
#         ROC.AUC(ROC.roc(yhat, y, truelabel))
#     return _auc
# end


## METHODS TO CLOSE GAP BETWEEN `Array`s, `DataFrame`s AND JuliaDB `Table`s

# note: for now trying to avoid complicated data interface to accommodate
# a variety of data containers but ultimately this may be unavoidable

select(df::AbstractDataFrame, selection) = df[selection]
(df::DataFrame)(rows) = view(df, rows)
(df::SubDataFrame)(rows) = view(df, rows)
(v::Vector)(rows) = v[rows]      # TODO might later try to return views but require extra complexity
(A::Matrix)(rows) = A[rows, :]
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
function Base.merge!(v1::Vector, v2::Vector)
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
        trainable.report = Dict{Symbol,Any}()

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
        state = trainable.state
    else
        state = nothing
    end
    args = (arg(rows) for arg in trainable.args)
    trainable.estimator, trainable.state, report =
        fit(trainable.model, args..., state, verbosity-1; kwargs...)
    if report != nothing
        merge!(trainable.report, report)
    end
    verbosity <1 || @info "Done."
    return trainable
end


## MODEL INTERFACE - SPECIFIC TO SUPERVISED LEARNERS

# users' trainable model constructor for supervised models:
function prefit(model::S, X=nothing, y=nothing) where S<:Supervised
    !(X == nothing || y == nothing) ||
        @warn "To make trainable instead use `prefit(model, X, y)` or later call `model.args = X, y`."
    return TrainableModel(model, X, y)
end

# predict method for learner models:
function predict(model::TrainableModel{L}, X) where L<: Learner 
    if isdefined(model, :estimator)
        return predict(model.model, model.estimator, X)
    else
        throw(error("$model is not trained and so cannot predict. Perhaps you meant to pass dynamic data?"))
    end
end

# TODO: predict_proba method for classifier models:


## MODEL INTERFACE - SPECIFIC TO UNSUPERVISED LEARNERS

# TODO


## MODEL INTERFACE - SPECIFIC TO TRANSFORMERS

# users' trainable model constructor for transformers:
function prefit(model::T, X=nothing) where T<:Transformer
    X != nothing ||
        @warn "To make trainable instead use `prefit(model, X)` or later call `model.args = X`."
    return TrainableModel(model, X)
end

function transform(model::TrainableModel{T}, X) where T<:Transformer
    if isdefined(model, :estimator)
        return transform(model.model, model.estimator, X)
    else
        throw(error("$model is not trained and so cannot transform. Perhaps you meant to pass dynamic data?"))
    end
end

function inverse_transform(model::TrainableModel{T}, X) where T<:Transformer
    if isdefined(model, :estimator)
        return inverse_transform(model.model, model.estimator, X)
    else
        throw(error("$model is not trained and so cannot inverse_transform.  Perhaps you meant to pass dynamic data?"))
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

# the "fit through" method:
function fit!(X::DynamicData, rows; verbosity=1, kwargs...)
    for trainable in X.tape[1:end-1]
        fit!(trainable, rows; verbosity=verbosity)
    end
    fit!(X.tape[end], rows; verbosity=verbosity, kwargs...)
    return X
end

    
## SYNTACTIC SUGAR FOR DYNAMIC DATA (PIPELINING)
    
dynamic(X) = dynamic(identity, X)  

predict(model::TrainableModel{L}, X::DynamicData) where L<:Learner =
    dynamic(predict, model, X)
transform(model::TrainableModel{T}, X::DynamicData) where T<:Transformer =
    dynamic(transform, model, X)
inverse_transform(model::TrainableModel{T}, X::DynamicData) where T<:Transformer =
    dynamic(inverse_transform, model, X)

array(X) = convert(Array, X)
array(X::DynamicData) = dynamic(array, X)

rms(y::DynamicData, yhat::DynamicData) = dynamic(rms, y, yhat)
rms(y, yhat::DynamicData) = dynamic(rms, y, yhat)
rms(y::DynamicData, yhat) = dynamic(rms, y, yhat)

import Base.+
+(y1::DynamicData, y2::DynamicData) = dynamic(+, y1, y2)
+(y1, y2::DynamicData) = dynamic(+, y1, y2)
+(y1::DynamicData, y2) = dynamic(+, y1, y2)


## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/KNN.jl")


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



    
