using Distributions
using StatsBase
using Tensors
using Iterators
using PyPlot
using MAT

percentage_error(result, value) = (result - value)/value

modify(data) = (size(data,1)-1)/size(data,1)

#     eig_val_vec(M) = eigvecs(M)[:,end:-1:1], diagm(sort(eigvals(M),  rev=true))
function eig_val_vec(M::Matrix) # zmienic na eig_vec_val
    F = eigfact(Symmetric(M))
    p = sortperm(F[:values], rev=true)
    F[:vectors][:,p], F[:values][p]
end

function cov_calc(data, n)
    X1, S = eig_val_vec(get_cumulant2(data))

    # test of the VAR method
    results = zeros(n,3)
    for i = 1:n
        linear_comb = data*X1[:,i]
        results[i,:] = [S[i], cov(linear_comb), percentage_error(S[i], cov(linear_comb))]
    end
    results, X1, S
end

function skewness_calc(data, n)
    HO = hosvd(get_cumulant3(data))
    X1 = HO.matrices[1]
    T3 = HO.coretensor

    results = zeros(n,3)
    for i = 1:n
        skew = skewness(data*X1[:,i])
        assymetry_tens = T3[i,i,i]/((modify(data)*cov(data*X1[:,i]))^(3/2))
        results[i,:] = [assymetry_tens, skew, percentage_error(assymetry_tens, skew)]
    end
    results, X1, T3
end

function kurtosis_calc(data, n)
    HO = hosvd(get_cumulant4(data))
    X1 = HO.matrices[1]
    T4 = HO.coretensor

    results = zeros(n,3)
    for i = 1:n
        kur = kurtosis(data*X1[:,i])
        kur_tens = T4[i,i,i,i]/((modify(data)*cov(data*X1[:,i]))^(2))
        results[i,:] = [kur_tens, kur, percentage_error(kur_tens, kur)]
    end
    results, X1, T4
end

function Extract_sk(A::Matrix, n)
    skew = zeros(n)
    kurt = zeros(n)
    for i =1:n
        skew[i] =  skewness(A[:,i])
        kurt[i] =  kurtosis(A[:,i])
    end
    skew, kurt
end

function Plot_AK1(As_array::Array, K_array::Array)
    plot(As_array, color="blue", linewidth=2.0, "bo", label="assymetry")
    plot(K_array, color="red", linewidth=2.0, "ro", label="kurtosis")
    title("Gaussian Distribution", fontsize = 12)
    axis([-1,21,-0.045, 0.025])
    xlabel("Number of sample", fontsize = 12)
    ylabel("Asymmetry and kurtosis", fontsize = 12)
    legend(fontsize = 12, loc = 3)
    savefig("figure1.pdf")
end

function Plot_AK2(As_array::Array, K_array::Array)
    plot(As_array, color="blue", linewidth=2.0, "bo", label="assymetry")
    plot(K_array, color="red", linewidth=2.0, "ro", label="kurtosis")
    title("Linear combination after HOSVD", fontsize = 12)
    axis([-1,21,-0.045, 0.025])
    xlabel("Number of sample", fontsize = 12)
    ylabel("Asymmetry and kurtosis", fontsize = 12)
    legend(fontsize = 12, loc = 3)
    savefig("figure2.pdf")
end
