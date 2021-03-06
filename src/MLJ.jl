module MLJ

export glossary
export Rows, Cols, Names
export features, X_and_y
export rms, rmsl, rmslp1, rmsp
export TrainableModel, Trainable, dynamic, fit!
export freeze!, thaw!
export array

# defined here but extended by files in "interfaces/" (lazily loaded)
export predict, transform, inverse_transform

# defined in include files:
export partition, @curve, @pcurve                  # "utilities.jl"
export @more, @constant                            # "show.jl"
export load_boston, load_ames, load_iris, datanow  # "datasets.jl"
export KNNRegressor                                # "builtins/KNN.jl":

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


import Requires.@require  # lazy code loading package
import CSV
import DataFrames: DataFrame, AbstractDataFrame, SubDataFrame, eltypes, names
import Distributions

# from Standard Library:
using Statistics
using LinearAlgebra


## CONSTANTS

const srcdir = dirname(@__FILE__) # the directory containing this file 
const TREE_INDENT = 2 # indentation for tree-based display of dynamic data 
const COLUMN_WIDTH = 24           # for displaying dictionaries with `show`
const DEFAULT_SHOW_DEPTH = 2      # how deep to display fields of `MLJType` objects


## GENERAL PURPOSE UTILITIES

include("utilities.jl")


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# overload `show` method for MLJType (which becomes the fall-back for
# all subtypes):
include("show.jl")

# for storing hyperparameters:
abstract type Model <: MLJType end 

abstract type Learner <: Model end

# a model type for transformers
abstract type Transformer <: Model end 

# special learners:
abstract type Supervised{E} <: Learner end # parameterized by fit-result `E`
abstract type Unsupervised{E} <: Learner end

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


## UNIVERSAL ADAPTOR FOR DATA CONTAINERS

# For vectors and tabular data containers `df`:
# `df[Rows, r]` gets rows of `df` at `r` (single integer, integer range, or colon)
# `df[Cols, c]` selects features of df at `c` (single integer or symbol, vector of symbols, integer range or colon); not supported for vectors
# `df[Names]` returns names of all features of `df` (or indices if unsupported)


struct Rows end
struct Cols end
struct Names end
struct Eltypes end
struct Echo end # needed to terminate calls of dynamic data types on unseen source data

Base.getindex(df::AbstractDataFrame, ::Type{Rows}, r) = df[r,:]
Base.getindex(df::AbstractDataFrame, ::Type{Cols}, c) = df[c]
Base.getindex(df::AbstractDataFrame, ::Type{Names}) = names(df)
Base.getindex(df::AbstractDataFrame, ::Type{Eltypes}) = eltypes(df)
Base.getindex(df::AbstractDataFrame, ::Type{Echo}, dg) = dg

# Base.getindex(df::JuliaDB.Table, ::Type{Rows}, r) = df[r]
# Base.getindex(df::JuliaDB.Table, ::Type{Cols}, c) = select(df, c)
# Base.getindex(df::JuliaDB.Table, ::Type{Names}) = getfields(typeof(df.columns.columns))
# Base.getindex(df::JuliaDB.Table, ::Type{Echo}, dg) = dg

Base.getindex(A::AbstractMatrix, ::Type{Rows}, r) = A[r,:]
Base.getindex(A::AbstractMatrix, ::Type{Cols}, c) = A[:,c]
Base.getindex(A::AbstractMatrix, ::Type{Names}) = 1:size(A, 2)
Base.getindex(A::AbstractMatrix{T}, ::Type{Eltypes}) where T = [T for j in 1:size(A, 2)]
Base.getindex(A::AbstractMatrix, ::Type{Echo}, B) = B

Base.getindex(v::AbstractVector, ::Type{Rows}, r) = v[r]
Base.getindex(v::AbstractVector, ::Type{Echo}, w) = w


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

features(task::Task) = filter!(task.data[Names]) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(task.data[Names]) do ftr
    ftr != task.target && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = (task.data[Cols, features(task)],
                                 task.data[Cols, task.target])


## SOME LOCALLY ARCHIVED TASKS FOR TESTING AND DEMONSTRATION

include("datasets.jl")


## LOW-LEVEL MODEL METHODS 

# Most concrete model types, and their associated low-level methods,
# are defined in package interfaces, located in
# "/src/interfaces.jl". These are are lazily loaded (see the end of
# this file). Built-in model definitions and associated methods (ie,
# ones not dependent on external packages) are contained in
# "/src/builtins.jl"

# low-level methods to be extended:
function fit end
function fit2 end
function predict end
function predict_proba end
function transform end 
function inverse_transform end

# fallback method to correct invalid hyperparameters and return
# a warning (in this case empty):
clean!(fitresult::Model) = ""

# fallback method for refitting:
fit2(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)


## TRAINABLE MODEL INTERFACE - BARE BONES

""" 
    merge!(tape1, tape2)

Incrementally appends to `tape1` all elements in `tape2`, excluding
any element previously added (or any element of `tape1` in its initial
state).

"""
function Base.merge!(tape1::Vector, tape2::Vector)
    for trainable in tape2
        if !(trainable in tape1)
            push!(tape1, trainable)
        end
    end
    return tape1
end

# TODO: replace linear tapes below with dependency trees to allow
# better scheduling of training dynamic data.

mutable struct TrainableModel{B<:Model} <: MLJType

    model::B
    fitresult
    cache
    args
    report
    tape::Vector{TrainableModel}
    frozen::Bool
    
    function TrainableModel{B}(model::B, args...) where B<:Model

        trainable = new{B}(model)
        trainable.frozen = false
        trainable.args = args
        trainable.report = Dict{Symbol,Any}()

        # note: `get_tape(arg)` returns arg.tape where this makes
        # sense and an empty tape otherwise.  However, the complete
        # definition of `get_tape` must be postponed until
        # `LearningNode` type is defined.

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

    if trainable.frozen && verbosity > -1
        @warn "$trainable not trained as it is frozen."
        return trainable
    end
        
    verbosity < 1 || @info "Training $trainable whose model is $(trainable.model)."

    args = (arg[Rows, rows] for arg in trainable.args)

    if !isdefined(trainable, :fitresult)
        trainable.fitresult, trainable.cache, report =
            fit(trainable.model, verbosity, args...)
    else
        trainable.fitresult, trainable.cache, report =
        fit2(trainable.model, verbosity, trainable.fitresult, trainable.cache, args...; kwargs...)
    end

    if report != nothing
        merge!(trainable.report, report)
    end

#    verbosity <1 || @info "Done."

    return trainable

end


## MODEL INTERFACE - SPECIFIC TO SUPERVISED LEARNERS

# users' trainable model constructor for supervised models:
function Trainable(model::S, X, y) where S<:Supervised
    return TrainableModel(model, X, y)
end

# predict method for learner models:
function predict(trainable::TrainableModel{L}, X) where L<: Learner 
    if isdefined(trainable, :fitresult)
        return predict(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot predict."))
    end
end

# TODO: predict_proba method for classifier models:


## MODEL INTERFACE - SPECIFIC TO UNSUPERVISED LEARNERS

# TODO


## MODEL INTERFACE - SPECIFIC TO TRANSFORMERS

# users' trainable model constructor for transformers:
function Trainable(model::T, X) where T<:Transformer
    return TrainableModel(model, X)
end

function transform(trainable::TrainableModel{T}, X) where T<:Transformer
    if isdefined(trainable, :fitresult)
        return transform(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot transform."))
    end
end

function inverse_transform(trainable::TrainableModel{T}, X) where T<:Transformer
    if isdefined(trainable, :fitresult)
        return inverse_transform(trainable.model, trainable.fitresult, X)
    else
        throw(error("$trainable with model $(trainable.model) is not trained and so cannot inverse_transform."))
    end
end


## DYNAMIC DATA INTERFACE - BASICS

struct LearningNode{M<:Union{TrainableModel, Nothing}} <: MLJType

    operator::Function              # eg, `predict` or `inverse_transform` or a static operator
    trainable::M                        # is `nothing` for static operators
    args                            # data (static or dynamic) furnishing inputs for `operator`
    tape::Vector{TrainableModel}    # for tracking dependencies
    depth::Int64       

    function LearningNode{M}(operator, trainable::M, args...) where M<:Union{TrainableModel, Nothing}

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

        depth = maximum(get_depth(arg) for arg in args) + 1

        return new{M}(operator, trainable, args, tape, depth)

    end
end

# ... where
get_depth(::Any) = 0
get_depth(X::LearningNode) = X.depth

# to complete the definition of `TrainableModel` and `LearningNode`
# constructors:
get_tape(::Any) = TrainableModel[]
get_tape(X::LearningNode) = X.tape
get_tape(trainable::TrainableModel) = trainable.tape

# autodetect type parameter:
LearningNode(operator, trainable::M, args...) where M<:Union{TrainableModel, Nothing} =
    LearningNode{M}(operator, trainable, args...)

# user's fall-back constructor:
dynamic(operator::Function, trainable::TrainableModel, args...) = LearningNode(operator, trainable, args...)

# user's constructor for learners and transformers:
LT = Union{Learner,Transformer}
function dynamic(operator::Function, trainable::TrainableModel{M}, args...) where M<:LT
    length(args) == 1 || throw(error("Wrong number of arguments. "*
                                     "Use `dynamic(operator, trainable_model, X)` for learners or transfomers."))
    return LearningNode(operator, trainable, args...)
end

# user's special constructor for static operators:
function dynamic(operator::Function, args...)
    length(args) > 0 || throw(error("`args` in `dynamic(::Function, args...)` must be non-empty. "))
    trainable = nothing
    return LearningNode(operator, trainable, args...)
end

# make dynamic data row-indexable:
Base.getindex(y::LearningNode, ::Type{Rows}, r) =
    (y.operator)(y.trainable, [y.args[j][Rows,r] for j in eachindex(y.args)]...)

# special case of static operators:
Base.getindex(y::LearningNode{Nothing}, ::Type{Rows}, r) =
    (y.operator)([y.args[j][Rows,r] for j in eachindex(y.args)]...)

# note: the following two methods only work as expected if the dynamic data `y`
# has a single ultimate source:

# calling `y[Echo, Xnew]` gets "value" of `y` as if ultimate source of
# `y` (as learning network) is replaced with `Xnew`:
Base.getindex(y::LearningNode, ::Type{Echo}, Xnew) =
    (y.operator)(y.trainable, [y.args[j][Echo,Xnew] for j in eachindex(y.args)]...)

# specializing to static operators:
Base.getindex(y::LearningNode{Nothing}, ::Type{Echo}, Xnew) =
    (y.operator)([y.args[j][Echo,Xnew] for j in eachindex(y.args)]...)

# the "fit through" method:
function fit!(y::LearningNode, rows; verbosity=1, kwargs...)
    for trainable in y.tape[1:end-1]
        fit!(trainable, rows; verbosity=verbosity)
    end
    fit!(y.tape[end], rows; verbosity=verbosity, kwargs...)
    return y
end

# overload show method:
function spaces(n)
    s = ""
    for i in 1:n
        s = string(s, " ")
    end
    return s
end
function Base.show(stream::IO, ::MIME"text/plain", X::LearningNode)
    gap = spaces(20 - TREE_INDENT*get_depth(X) + TREE_INDENT)
    if X.operator == identity && !(X.args[1] isa LearningNode) 
        print(stream, gap, handle(X))#.args[1]))
    else
        detail = (X.trainable == nothing ? "(" : "($(handle(X.trainable)),")
        operator_name = typeof(X.operator).name.mt.name
        #    println(stream, gap, handle(X), " = ", operator_name, detail)
        println(stream, gap, operator_name, detail)
        n_args = length(X.args)
        counter = 1
        for arg in X.args
            if arg isa LearningNode
                show(stream, MIME("text/plain"), arg)
            else
                # id = objectid(arg)
                # if id in keys(handle_given_id)
                #     representation = handle_given_id[id]
                # else
                #     representation = "*"
                # end
                # print(stream, gap[1:end-TREE_INDENT], representation)
                print(stream, gap, spaces(TREE_INDENT), handle(arg))
            end
            counter >= n_args || println(stream, ", ")
            counter += 1
        end
    print(stream, ")")
    end
end


## SYNTACTIC SUGAR FOR DYNAMIC DATA
    
dynamic(X) = dynamic(identity, X)
dynamic(X::LearningNode) = X

# make dynamic data callable on unseen source data:
(y::LearningNode)(Xnew) = y[Echo, Xnew]

# TODO: write method `source` that locates ultimate source of a dynamic
# data's input data

# remove need for `dynamic` syntax in case of operators of main interest:
predict(trainable::TrainableModel{L}, X::LearningNode) where L<:Learner =
    dynamic(predict, trainable, X)
transform(trainable::TrainableModel{T}, X::LearningNode) where T<:Transformer =
    dynamic(transform, trainable, X)
inverse_transform(trainable::TrainableModel{T}, X::LearningNode) where T<:Transformer =
    dynamic(inverse_transform, trainable, X)

array(X) = convert(Array, X)
array(X::LearningNode) = dynamic(array, X)

Base.log(v::Vector{<:Number}) = log.(v)
Base.exp(v::Vector{<:Number}) = exp.(v)
Base.log(X::LearningNode) = dynamic(log, X)
Base.exp(X::LearningNode) = dynamic(exp, X)


rms(y::LearningNode, yhat::LearningNode) = dynamic(rms, y, yhat)
rms(y, yhat::LearningNode) = dynamic(rms, y, yhat)
rms(y::LearningNode, yhat) = dynamic(rms, y, yhat)

import Base.+
+(y1::LearningNode, y2::LearningNode) = dynamic(+, y1, y2)
+(y1, y2::LearningNode) = dynamic(+, y1, y2)
+(y1::LearningNode, y2) = dynamic(+, y1, y2)


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
    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=false
end

end # module



    
