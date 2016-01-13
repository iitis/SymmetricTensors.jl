using Distributions
using StatsBase

function generate_random_gauss(M, L)
    D = rand(M, M)
    C = D*D'
    data = rand(MvNormal(C), L)', M # ugh?
end
