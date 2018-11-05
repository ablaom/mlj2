# Dynamic data (aka learning networks)

Warning: The syntax shown here is out-dated and the description of
composite learners a little over-simplified.

Dynamic data behaves superficially like regular data (e.g., a data
frame) but tracks its dependencies on other data (static and dynamic),
as well as the training events that were used to define them. You can
think of dynamic data as nodes in a learning network if you want to.

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
julia> scale = Trainable(scale_, X)
julia> fit!(scale, fold1)
[ Info: Trainable TrainableModel @ ...170.
[ Info: Done.
````

Note here that training is split into two phases: a "prefit" stage, in
which hyperparameters are wrapped in *all* of the data (to create a
`TrainableModel` object) but not told which part (rows) of the data is
for training; and a final training stage, in which we declare which
part of the data we want to use.

````julia
julia> # get the transformed inputs:
julia> Xt = transform(scale, X);

julia> # convert data frame `Xt` to an array:
julia> Xa = array(Xt);

julia> # choose a learner and train it on the same fold:
julia> knn_ = KNNRegressor(K=7) # just a container for hyperparameters
julia> knn = Trainable(knn_, Xa, y)
julia> fit!(knn, fold1)
[ Info: Trainable TrainableModel @ ...838.
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
julia> scale = Trainable(scale_, X) # no need to train!

julia> # get the transformed inputs, as if `scale` were already trained:
julia> Xt = transform(scale, X)

julia> # convert DataFrame Xt to an array:
julia> Xa = array(Xt)

julia> # choose a learner and make it trainable:
julia> knn_ = KNNRegressor(K=7)
julia> knn = Trainable(knn_, Xa, y) # no need to train!

julia> # get the predictions, as if `knn` already trained:
julia> yhat = predict(knn, Xa)

julia> # compute the error:
julia> er = rms(y, yhat)

````

Now `er` is dynamic, so we can do "look-through" training on any rows we
like and evaluate on any rows we like. Look-through training means the scalar and KNN get refitted automatically:

````julia
julia> fit!(er, fold1)
[ Info: Trainable TrainableModel @ ...940.
[ Info: Done.
[ Info: Trainable TrainableModel @ ...251.
[ Info: Done.

julia> er1 = er(fold2)
7.32782969364479

julia> fit!(er, fold2)
[ Info: Trainable TrainableModel @ ...940.
[ Info: Done.
[ Info: Trainable TrainableModel @ ...251.
[ Info: Done.

julia> er2 = er(fold1)
9.616116727127416

julia> er = (er1 + er2)/2
8.471973210386103
````

In fact one can export the learning network as a stand alone model as
follows.  First we wrap our specification of the network in a function
whose arguments are X and y (we can drop the dynamic conversions,
which will be understood):

````julia 
function f(X,y) scale_ = Standardizer() 
    scale = Trainable(scale_, X) 
    Xt = transform(scale, X) 
    Xa = array(Xt) knn_ = KNNRegressor(K=7) 
    knn = Trainable(knn_, Xa, y) 
    yhat = predict(knn, Xa)
    return yhat 
end 
```` 

Secondly, there will be a model type called `CompositeLearner` which
has a single hyperparameter called `plumbing`.  We instantiate our
composite model with

````julia
composite = CompositeLearner(plumbing=f)
````

In MLJ we must write fit and predict methods for objects of type
`CompositeLearner` (omitted here). Once this is done the model
`composite` behaves like any other. 

