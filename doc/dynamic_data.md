I have made some encouraging progress on how to design learning
networks and want to report this progress here. My solution is
inspired by Mike Innes' work on Flux.  **The idea is that you just
want to write down the math, and have the framework wrap this in the
appropriate logic under the hood.** See also, the post, [On Machine
Learning and Programming
Languages](https://julialang.org/blog/2017/12/ml&pl).

I will formulate my solution in terms of "dynamic data". Dynamic data
behaves superficially like regular data (e.g., a data frame) but
tracks its dependencies on other data (static and dynamic), as well as
the training events that were used to define them. You can think of
dynamic data as nodes in a learning network if you want to, but the
average user probably doesn't care.

The dynamic data type and "trainable model" type (different from the
current MLJ one) are interdependent and must be defined in just the
right way to make it all work. I think I have it now. Below is a
preview of the syntax from a working implementation (from a private
repo).  I will discuss details elsewhere.


### Dynamic data and look-through training 

A.k.a. Learning pipelines/networks

Let's get some data (the Boston data set):

````julia
julia> using MLJ
julia> X, y = datanow(); # ALL of the data, training, test and validation

julia> # split the rows into training and testing rows:
julia> fold1, fold2 = partition(eachindex(y), 0.7) # 70:30 split
([1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  345, 346, 347, 348, 349, 350, 351, 352, 353, 354], [355, 356, 357, 358, 359, 360, 361, 362, 363, 364  …  497, 498, 499, 500, 501, 502, 503, 504, 505, 506])
````

#### Cross-validation the hard way

````julia
julia> # construct a transformer to standardize the inputs, using the
julia> # training fold to prevent data leakage:
julia> scale_ = Standardizer()
julia> scale = prefit(scale_, X)
julia> fit!(scale, fold1)
[ Info: Training TrainableModel @ ...170.
[ Info: Done.
````

Note here that training is split into two phases: a *prefit* stage, in
which hyperparameters are wrapped in *all* of the data, but not told
which part (rows) of the data is for training; and a final training
stage, in which we declare which part of the data we want to use. This
is slightly more complicated than the standard approach but critical
to the dynamic approach described later.

````julia
julia> # get the transformed inputs:
julia> Xt = transform(scale, X);

julia> # convert data frame `Xt` to an array:
julia> Xa = array(Xt);

julia> # choose a learner and train it on the same fold:
julia> knn_ = KNNRegressor(K=7) # just a container for hyperparameters
julia> knn = prefit(knn_, Xa, y)
julia> fit!(knn, fold1)
[ Info: Training TrainableModel @ ...838.
[ Info: Done.

julia> # get the predictions on the other fold:
julia> yhat = predict(knn, Xa(fold2));

julia> # compute the error:
julia> er1 = rms(y(fold2), yhat)
7.32782969364479
````

Then we must repeat all of the above with roles of `fold1` and `fold2`
reversed to get `er2` (omitted).

And then average `er1` and `er2` to get our estimate of the
generalization error.


#### Cross-validation using dynamic data and look-through training:

We will need two lines of code not used above, but everything else
will be easier and use almost identical syntax:

````julia
julia> X = dynamic(X)
julia> y = dynamic(y)

julia> # construct a transformer to standardize the inputs:
julia> scale_ = Standardizer()
julia> scale = prefit(scale_, X) # no need to train!

julia> # get the transformed inputs, as if `scale` were already trained:
julia> Xt = transform(scale, X)

julia> # convert DataFrame Xt to an array:
julia> Xa = array(Xt)

julia> # choose a learner and make it trainable:
julia> knn_ = KNNRegressor(K=7)
julia> knn = prefit(knn_, Xa, y) # no need to train!

julia> # get the predictions, as if `knn` already trained:
julia> yhat = predict(knn, Xa)

julia> # compute the error:
julia> er = rms(y, yhat)

````

Now `er` is dynamic, so we can do "look-through" training on any rows we
like and evaluate on any rows we like. Look-through training means the scalar and KNN get refitted automatically:

````julia
julia> fit!(er, fold1)
[ Info: Training TrainableModel @ ...940.
[ Info: Done.
[ Info: Training TrainableModel @ ...251.
[ Info: Done.

julia> er1 = er(fold2)
7.32782969364479

julia> fit!(er, fold2)
[ Info: Training TrainableModel @ ...940.
[ Info: Done.
[ Info: Training TrainableModel @ ...251.
[ Info: Done.

julia> er2 = er(fold1)
9.616116727127416

julia> er = (er1 + er2)/2
8.471973210386103
````

---

Thanks for raising those good points. Let me respond to the more
serious concerns for now.

In fact one can export the learning network as a stand alone model as
follows.  First we wrap our specification of the network in a function
whose arguments are X and y (we can drop the dynamic conversions,
which will be understood):

````julia 
function f(X,y) scale_ = Standardizer() scale =
prefit(scale_, X) Xt = transform(scale, X) Xa = array(Xt) knn_ =
KNNRegressor(K=7) knn = prefit(knn_, Xa, y) yhat = predict(knn, Xa)
return yhat end ```` Secondly, there will be a model type called
`CompositeLearner` which has a single hyperparameter called `plumbing`
(for me "model" is just a container for hyperparameters).  We
instantiate our composite model with

````julia
composite = CompositeLearner(plumbing=f)
````

In MLJ we must write fit and predict methods for objects of type
`CompositeLearner` but these are one-liners using the established
syntax. There is nothing else here the user needs to do. The model
`composite` behaves like any other. You can give it data when you're
ready, or play with the task API or whatever. So the impact of my
proposal on the rest of the interface is nil.

It was my ambition to accommodate learning networks of arbitrary
complexity. Basically any data manipulation can be accommodated (after
overloading for the dynamic data type). However, for simple
*sequential* networks the user could opt out of the construction as
given here. For example, they could compose models directly, as in

````julia
composite = transformer * learner
````

And we can implement `*` (and other stuff) in MLJ using the proposed
syntax with mimimal effort:

````julia
function *(transformer,learner)
    function f(X, y)
        transformer_ = prefit(transformer,X)
        Xt =  transform(transformer_, X)
        learner_ = prefit(learner, Xt, y)
        return predict(learner_, Xt)
    end
    return CompositeLearner(plumbing=f)
end
````

Incidentally, I imagined MLJ would want to give users the option of
constructing and fitting their own models directly, ie, without the
extra layer of abstraction of tasks, etc. By providing the "arcane"
fit-predict syntax, a scikit-learn user coming to MLJ can hit the
ground running, learning the other stuff as his demands for automation
increase. What I am offering, in exchange for provision of this syntax
(which the user could choose to ignore) is a markup-language for
complex learning networks, essentially for free.

---

### Stacking via dynamic data

Here's ho
