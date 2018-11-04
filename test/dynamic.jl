module TestDynamic

using Revise
using Test
using MLJ


# TRAINABLE MODELS

X_frame, y = datanow(); # boston data
X = array(X_frame)

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y)
train, valid, test = partition(allrows, 0.7, 0.15)
@test vcat(train, valid, test) == allrows
knn1 = Trainable(knn_, X, y)
fit!(knn1, train);
the_error = rms(predict(knn1, X[Rows, test]), y[test])

# TODO: compare to constant regressor and check it's significantly better


## DYNAMIC DATA

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn1))

X, y = datanow(); # ALL of the data, training, test and validation

# split the rows into training and testing rows:
allrows = eachindex(y)
fold1, fold2 = partition(allrows, 0.7) # 70:30 split

@constant Xd = dynamic(X)
@constant yd = dynamic(y)

# construct a transformer to standardize the target:
uscale_ = UnivariateStandardizer()
@constant uscale = Trainable(uscale_, yd)

# get the transformed inputs, as if `uscale` were already fit:
@constant z = transform(uscale, yd)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
@constant scale = Trainable(scale_, Xd) # no need to fit

# get the transformed inputs, as if `scale` were already fit:
@constant Xt = transform(scale, Xd)

# convert DataFrame Xt to an array:
@constant Xa = array(Xt)

# choose a learner and make it trainable:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
@constant knn = Trainable(knn_, Xa, z) # no need to fit

# get the predictions, as if `knn` already fit:
@constant zhat = predict(knn, Xa)

# inverse transform the target:
@constant yhat = inverse_transform(uscale, zhat)

# compute the error:
er = rms(yd, yhat)

# Now `er` is dynamic, so we can do "look through training" on any rows
# we like and evaluate on any rows:
fit!(er, fold1)
er[Rows, fold2]

fit!(er, fold2)
er_number = er[Rows, fold1]

# test echo:
Echo = MLJ.Echo
Xnew, ynew = datanow();
Xnew = Xnew[Rows, 3:7];
ynew = ynew[3:7];
@test X[Echo, Xnew] == Xnew
@test Xt[Echo, Xnew] == transform(scale, Xnew)
@test Xa[Echo, Xnew] == array(transform(scale, Xnew))
@test yhat[Echo, Xnew] == inverse_transform(uscale,
                        predict(knn, array(transform(scale, Xnew))))
@test yhat(Xnew) == yhat[Echo, Xnew]

end
