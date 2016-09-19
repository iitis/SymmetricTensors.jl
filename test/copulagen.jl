using Distributions
# generates random multivariate data using Clayton copula
function clcopulagen(t::Int, m::Int)
    theta = 1.02
    coredist = Gamma(1,1/theta)
    x = rand(t)
    u = rand(t,m)
    ret = zeros(t,m)
    invphi(x::Array{Float64,1}, theta::Float64) = (1+ theta.*x).^(-1/theta)
    for i = 1:m
        uniform = invphi(-log(u[:,i])./quantile(coredist, x), theta)
        ret[:,i] = quantile(Weibull(1+0.01*i,1), uniform)
    end
    ret
end

