"""Load a well-known public regression dataset with nominal features."""
function load_boston()
    df = CSV.read(joinpath(srcdir, "datanow", "Boston.csv"),
                  categorical=false, allowmissing=:none)
    return RegressionTask(data=df, target=:MedV, ignore=[:Chas]) 
end

"""Load a reduced version of the well-known Ames Housing task,
having six numerical and six categorical features."""
function load_ames()
    df = CSV.read(joinpath(srcdir, "datanow", "reduced_ames.csv"),
                  categorical=false, allowmissing=:none)
    df[:target] = exp.(df[:target])
    return RegressionTask(data=df, target=:target) 
end

"""Load a well-known public classification task with nominal features."""
function load_iris()
    df = CSV.read(joinpath(srcdir, "datanow", "iris.csv"),
                  categorical=false, allowmissing=:none)
    return ClassificationTask(data=df, target=:target)
end
