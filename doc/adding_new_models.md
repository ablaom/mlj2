# Guide for adding new models to MLJ 

This guide outlines the specification for the lowest level of the MLJ
application interface. It is is guide for those adding new models by
(i) writing glue code for lazily loaded external packages (the main
case); and (ii) writing code that is directly included in MLJ in the
form of an include file.

A checklist for adding models is given at the end, and a template for
adding supervised learner models from external packages is at
["src/interfaces/DecisionTree.jl"](../src/interfaces/DecisionTree.jl)


## Preliminaries


### MLJ types

Every type introduced the core MLJ package should be a subtype of:

```
abstract type MLJType end
```

The Julia `show` method is informatively overloaded for this
type. Variable bindings declared with `@constant` "register" the
binding, which is reflected in the output of `show`.


### Models

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
`Transformer`. 

> Later we may introduce other types for composite "learners" that do
> things not fitting within the existing framework.

## Package interfaces (glue code)

Note: Most of the following remarks also apply to built-in learning
algorithms (i.e., not defined in external packages) and presently
located in "src/builtins/". In particular "src/transforms.jl" will
contain a number of common preprocessing transformations, both "static" and
learned. External package interfaces go in "src/interfaces/"

Every package interface should live inside a submodule for namespace
hygene (see the template at
"src/interfaces/DecisionTree.jl"). Ideally, package interfaces should
export no `struct`s outside of the model types they define, and import
only abstract types. All "structural" design should be restricted to
the MLJ core to prevent rewriting glue code if (when!) we change our
design.

### New model type declarations

Here is an example of a concrete model type declaration:

````julia

R = Tuple{Matrix{Float64},Vector{Float64}}

mutable struct KNNRegressor <: Regressor{R}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function # each target value is weighted by `kernel(distance^2)`
end

````

Models (which are mutable) should never have internally defined
constructors but should always be given an external lazy keyword
constructor of the same name that defines default values and checks
their validity by calling an optional clean! method (which has a
trivial fall-back). 


### Supervised models

For every concrete type `ConcreteModel{R} <: Supervised{R}` a number
of "basement-level" methods are defined. These are what go into
package interfaces, together with model declerations.


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
(eg, add decision trees to a random forest). **Important:** Instructions to perform
special training functions must be passed as `kwargs`, which are their
exclusive purpose. Any other "instructions" (e.g., train using 5 threads)
must be conveyed by the hyperparameter values (fields of `learner`).

> By passing instructions to fit in this way avoids having to
> explicitly export package-specific methods to higher levels of the
> interface. It is easy to pass kwargs given to a higher level fit
> method down into the lower level method without the higher level
> needing to know anything about them.

**Iterative algorithims.** To enhance functionality at higher levels of the interface, the
`kwargs` instruction for adding `n` iterations to an iterative method
must be `add=n`.

Note that `fit` must accept `nothing` as a valid input `state` but
should never *return* `nothing` as `state` output. For the majority of
supervised learners, the fit-result serves well enough as the output
`state`.

The Julia type of `state` is not presently restricted.

Another possible use case for special fit instructions is the
composite learner. For example, we may want to retrain a composite of
an ordinary learner with a scaling transformer without refitting the
scaling when tuning the ordinary learner's hyperparameters. This
requires that the previous state of the composite be passed in to the
composite's fit method.

> One issue here is whether passing fit with special instructions
> might have unpredictable consequences if *new* data is also passed.

Any training-related statistics (e.g., internal estimate of the
generalization error, feature importances), as well as the results of
learner-specific functionality requested by `kwargs`, should be
returned in the `report` object, which is either a `Dict{Symbol,Any}`
object, or `nothing` if there is nothing to report. So for example,
`fit` might declare `report[:feature_importances]=...`.  Reports get
merged with those generated by previous calls to `fit` at higher
levels of the MLJ interface.

The types of the training data `X` and `y` should be whatever is
required by the package for the training algorithm and declared in the
`fit` type signature and documentation for safety.  Checks not
specific to the package (e.g., dimension matching checks) should be
left to higher levels of the interface to avoid code duplication.

**Hyperparameter checks.** The method `fit` should initially call
`clean!` on `learner` and issue the returned warning if changes are
made (i.e., the message is non-empty). See the template for an
example. **This is the only time `fit` should alter hyperparameter
values.** If the package is able to suggest better hyperparameters, as
part of training, return these in the report field. 

The `verbosity` level (0 for silent) is for passing to the fit method
of the external package. Package interfaces should generally avoid any
logging to avoid duplication at higher levels of the interface.

**Mutation of `fit` arguments.** The fit method should not alter any
of its arguments with the exception of `state` (which would be
discarded by a higher-level calling function in favour of the new
state). A normal `fit` call (i.e., default `kwargs`) should not make
use of `state`. 

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

**Iterative methods.** Iterative methods which can restart training by
giving `fit` the keyword instruction `add=n` should also declare

````julia
isiterative(learner::ConcreteModel) = true
````

to enable the enhanced iterative method functionality at higher levels
of the interface.

##  Checklist for new adding models 

At present the checklist is just for supervised learner models in
lazily loaded external packages.

1. Copy and edit file ["src/DecisionTree.jl"](../src/DecisionTree.jl)
which is annotated for use as a template. Give your new file a name
identical to the package name, including ".jl" extension, such as
"DecisionTree.jl". Put this file in "src/interfaces/".

2. Register your package for lazy loading with MLJ by finding out the
UUID of the package and adding an appropriate line to the `__init__`
method at the end of "src/MLJ.jl". It will look something like this:

````julia
function __init__()
   @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true
   @load_interface NewExternalPackage "893749-98374-9234-91324-1324-9134-98" lazy=true
end
````

With `lazy=true`, your glue code only gets loaded by the MLJ user
after they run 'import NewExternalPackage'. For testing in your local
MLJ fork, you may want to set `lazy=false` but to use `Revise` you
will also need to move the `@load_interface` line out outside of the
`__init__` function. 

3. Write self-contained test-code for the methods defined in your glue
code, in a file with an identical name, but placed in "test/", eg,
["test/DecisionTree.jl"](../test/DecisionTree.jl) for an example. This
code should be wrapped in a module to prevent namespace conflicts with
other test code. For a module name, just prepend "Test", as in
"TestDecisionTree". See "test/DecisionTree.jl" for an example. 

4. Add a line to ["test/runtests.jl"](../test/runtests.jl) to
`include` your test file, for the purpose of testing MLJ core and all
currently supported packages, including yours. You can Test your code
by running `test MLJ` from the Julia interactive package manager. You
will need to `dev` your local MLJ fork first. To test your code in
isolation, locally edit "test/runtest.jl" appropriately.

4. Make a pull-request to include your working inteface!
