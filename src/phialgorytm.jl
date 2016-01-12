#Further
using Distributions
using StatsBase
using Tensors
using MAT


multiply_mode(n, T, M) = Tensors.modemult_list(T, enumerate(repeated(M, n)))


X = eye(n)

function finding(X)
    X = eigvecs(X*cumulantT2*X')[:,end:-1:1]
    X = hosvd(multiply_mode(3, cumulantT3, X')).matrices[1]
    X = hosvd(multiply_mode(4, cumulantT4, X')).matrices[1]
    X
end

function finding_last_step(X)
    X = eigvecs(X*cumulantT2*X')[:,end:-1:1]
    Sigma = diagm(sort(eigvals(X*cumulantT2*X'),  rev=true))
    HO = hosvd(multiply_mode(3, cumulantT3, X'))
    X = HO.matrices[1]
    T3 = HO.coretensor
    HO = hosvd(multiply_mode(4, cumulantT4, X'))
    X = HO.matrices[1]
    T4 = HO.coretensor
    X, Sigma, T3, T4
end


for i = 1:100000
    X = finding(X)
end


X, C2, C3, C4 = finding_last_step(X);
