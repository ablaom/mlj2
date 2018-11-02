### MLJ types

Every type introduced the core MLJ package should be a subtype of:

```
abstract type MLJType end
```

The Julia `show` method is overloaded for this type. Variable bindings
declared with `@constant` "register" the binding, which is reflected
in the output of `show`.


### The model interface

A *model* is an object storing hyperparameters associated with some
machine learning algorithm, where "learning algorithm" is to be
broadly interpreted.  The name of the Julia type associated with a
model indicates the associated algorithm (e.g.,
`DecisionTreeClassifier`). The ultimate supertype of all models is:

````julia
abstract type Model <: MLJType end 
````

Informally, we divide learning algorithms into those intended for
making "predictions", called *learners* (e.g., the CART decision tree
algorithm) and those intended for "transforming" data (based on
previously seen data), called *transformers* (e.g., the PCA
feature-reduction algorithm).  Generally, only transformers convert
data in two directions (can *inverse* transform) and only supervised
learners have more input variables for training than for prediction
(but the distinction might otherwise be vague). We use the same words,
*learner* and *transformer*, for the *models* associated with these
algorithms.

By a *fit-result* we shall mean an object storing the "weights" or
"paramaters" learned by an algorithm using the hyperparameters
specified by a model (i.e., what a learner needs to predict or what a
transformer needs to transform). There is no abstract type for
fit-results because these types are generally declared in external
packages. However, in MLJ superivised learners are parametrized by their
fit-result type `R`, for efficient implementation of large
ensembles of learners of uniform type.

````julia
abstract type Learner <: Model end
    abstract type Supervised{R} <: Learner end
	    abstract type Regressor{R} <: Supervised{R} end
		abstract type Classifier{R} <: Supervised{R} end

    abstract type Unsupervised <: Learner end

abstract type Transformer <: Model end 
````

Presently, every concrete model type in MLJ must be an immediate
subtype of `Regressor{R}`, `Classifier{R}`, `Unsupervised` or
`Transformer`. Here is an example of a concrete model type
declaration:

````julia

R = Tuple{Matrix{Float64},Vector{Float64}}

mutable struct KNNRegressor <: Regressor{R}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function # each target value is weighted by `kernel(distance^2)`
    frozen::Bool
end

````

Models (which are mutable) should never have internally defined
constructors but should always be given an external lazy keyword
constructor of the same name that defines default values. See
"src/interfaces/DecisionTree.jl" for a template. Checking correcting
for the validity of field values (hyperparameters) is performed by an
optional `clean!` method (see below).

The last field `frozen` is a compulsory field that is not actually a
hyperparameter, but it is needed to flag models that should not be
retrained in learning networks constructed at a higher level of the
interface. It's default value should be `false`.

### Supervised models

For every concrete type `ConcreteModel{R} <: Supervised{R}` a number
of "basement-level" methods are defined. 


#### Compulsory methods

````julia
result::R, state, report = 
    fit(learner::ConcreteModel, X, y, state, verbosity::Int; kwargs...)
````

Here `result` is the fit-result. Also returned is the `state`, which
means information sufficient to perform (on passing `state` as an
argument in a subsequent call to `fit`) any implemented functions
*specific to the learner type*, beyond normal training. This could be,
for example, pruning an existing decision tree, or optimizing weights
of an ensemble learner. In particular, in the case of iterative
learners, `state` must be sufficient to restart the training algorithm
(eg, add decision trees to a random forest). Instructions to perform
special training functions must be passed as `kwargs`, which are their
exclusive purpose. Any other "instructions" (e.g., train using 5 threads)
must be conveyed by the hyperparameter values (fields of `learner`).

Note that `fit` must accept `nothing` as a valid input `state` but
should never *return* `nothing` as `state` output. For the majority of
supervised learners, the fit-result serves well enough as the output
`state`.

To enhance functionality at higher levels of the interface, the
`kwargs` instruction for adding `n` iterations to an iterative method
must be `add=n`.

The Julia type of `state` is not presently restricted.

Any training-related statistics (e.g., internal estimate of the
generalization error, feature importances), as well as the results of
learner-specific functionality requested by `kwargs`, should be
returned in the `report` object, which is either a `Dict{Symbol,Any}`
object, or `nothing` if there is nothing to report. So for example,
`fit` might declare `report[:feature_importances]=...`.  Reports get
merged with those generated by previous calls to `fit` at higher
levels of the MLJ interface.

The types of the training data `X` and `y` should be whatever is most
natural for the training algorithm and declared in the `fit` type
signature and documentation for safety.

The method `fit` should initially call `clean!` on `learner` and issue
the returned warning if changes are made (the message is
non-empty). See the template.

The `verbosity` level (0 for silent) is for passing to the fit method
of the external package. Package interfaces should generally avoid any
logging to avoid duplication at higher levels of the interface.

> Presently, MLJ has a thin wrapper for fit-results called `ModelFit`
> and the output of `fit` in package interfaces is of this type. I
> suggest any such wrapping occur *outside* of package interfaces, to
> avoid rewriting them if the core design changes. For the same
> reason, I suggest that package interfaces import as little as
> possible from core.

````julia
prediction = predict(learner::ConcreteModel, result, Xnew)
````

Here `Xnew` should be the same type as `X` in the `fit` method. (So to
get a prediction on a single pattern, a user may need to suitably wrap
the pattern before passing to `predict` - as a single-row `DataFrame`,
for example - and suitably unwrap `prediction`, which must have the
same type as `y` in the `fit` method.)

#### Optional methods

A learner of `Classifier` type can implement a `predict_proba` method
to predict probabilities instead of labels, and will have the same
type signature as `predict` for its inputs.

````julia
message::String = clean!(learner::Supervised)
````

Checks and corrects for invalid fields (hyperparameters), returning a
warning `messsage`. Should only throw an exception as a last resort.

Iterative methods which can restart training by giving `fit` the
keyword instruction `add=n` should declare

````julia
isiterative(learner::ConcreteModel) = true
````

to enable the enhanced iterative method functionality at higher levels
of the interface.


