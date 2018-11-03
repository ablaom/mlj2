module TestTransformer

using MLJ
using Test

# relabelling with integer transformer:
y = rand(Char, 50)
allrows = eachindex(y)
test = 3:37
to_int_hypers = ToIntTransformer()
to_int = prefit(to_int_hypers, y)
fit!(to_int, allrows)
z = transform(to_int, y[test])
@test y[test] == inverse_transform(to_int, z)
to_int_hypers.map_unseen_to_minus_one = true
to_int = prefit(to_int_hypers, [1,2,3,4])
fit!(to_int, [1,2,3,4])
@test transform(to_int, 5) == -1
@test transform(to_int, [5,1])[1] == -1 

# `UnivariateStandardizer`:
stand = prefit(UnivariateStandardizer(), [0, 2, 4])
fit!(stand, 1:3)
@test round.(Int, transform(stand, [0,4,8])) == [-1.0,1.0,3.0]
@test round.(Int, inverse_transform(stand, [-1, 1, 3])) == [0, 4, 8] 
X, y = X_and_y(load_ames());
train, test = partition(eachindex(y), 0.9);

# introduce a field of type `Char`:
X[:OverallQual] = map(Char, X[:OverallQual]);

# `Standardizer`:
stand = prefit(Standardizer(), X)
fit!(stand, eachindex(y))
transform(stand, X)

stand = prefit(Standardizer(features=[:GrLivArea]), X)
fit!(stand, eachindex(y))
transform(stand, X)

end
