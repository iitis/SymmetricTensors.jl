get_cumulant1(data_matrix::Matrix) = Float64[mean(data_matrix[:,i]) for i = 1:n]

get_cumulant2(data_matrix::Matrix) = (cov(data_matrix, corrected = false))


function get_cumulant3{T<:AbstractFloat}(data_matrix::Matrix{T})
    centred = zeros(data_matrix)
    n = size(data_matrix, 2)
    for i = 1:n
        centred[:,i] = data_matrix[:,i]-mean(data_matrix[:,i])
    end
    cumulantT3 = SharedArray(T,n,n,n)

    @sync @parallel for i = 1:n
        for j = i:n, k = j:n
            a = mean(centred[:,i].*centred[:,j].*centred[:,k])
            cumulantT3[i,j,k] = a
            cumulantT3[i,k,j] = a
            cumulantT3[j,i,k] = a
            cumulantT3[j,k,i] = a
            cumulantT3[k,i,j] = a
            cumulantT3[k,j,i] = a
#             for p = permutations((i,j,k))
#                 cumulantT3[p] = m
#             end
        end
    end
    return Array(cumulantT3)
end

function get_cumulant4{T<:AbstractFloat}(data_matrix::Matrix{T})
    #auxiliary functions for cumulant 4
    n = size(data_matrix, 2)
    E(A,B,C,D) = mean(A.*B.*C.*D)

    E31(M1,M2, M3,N) = mean(M1.*M2.*M3)*mean(N)
    p31(A,B,C,D) = E31(A,B,C,D)+E31(A,B,D,C)+E31(A,D,C,B)+E31(D,B,C,A)

    E22(M1,M2, N1,N2) = mean(M1.*M2)*mean(N1.*N2)
    p22(A,B,C,D) = E22(A,B,C,D)+E22(A,C,B,D)+E22(A,D,C,B)

    E211(M1,M2, N,O) = mean(M1.*M2)*mean(N)*mean(O)
    p211(A,B,C,D) = E211(A,B,C,D)+E211(A,C,B,D)+E211(A,D,C,B)+E211(C,B,A,D)+E211(C,D,A,B)+E211(B,D,A,C)

    mixed_elements(A,B,C,D) = -p31(A,B,C,D)-p22(A,B,C,D)+2*p211(A,B,C,D)
    cumulant4(A,B,C,D) = (E(A,B,C,D)+mixed_elements(A,B,C,D)-6*mean(A)*mean(B)*mean(C)*mean(D))

    cumulantT4 = SharedArray(T,n,n,n,n)

    @sync @parallel for i = 1:n
        for j = i:n, k = j:n, l = k:n

            a = cumulant4(data_matrix[:,i], data_matrix[:,j], data_matrix[:,k], data_matrix[:,l])
            cumulantT4[i,j,k,l] = a
            cumulantT4[l,j,k,i] = a
            cumulantT4[i,l,k,j] = a
            cumulantT4[i,j,l,k] = a

            cumulantT4[i,k,j,l] = a
            cumulantT4[l,k,j,i] = a
            cumulantT4[i,l,j,k] = a
            cumulantT4[i,k,l,j] = a

            cumulantT4[j,i,k,l] = a
            cumulantT4[l,i,k,j] = a
            cumulantT4[j,l,k,i] = a
            cumulantT4[j,i,l,k] = a

            cumulantT4[j,k,i,l] = a
            cumulantT4[l,k,i,j] = a
            cumulantT4[j,l,i,k] = a
            cumulantT4[j,k,l,i] = a

            cumulantT4[k,i,j,l] = a
            cumulantT4[l,i,j,k] = a
            cumulantT4[k,l,j,i] = a
            cumulantT4[k,i,l,j] = a


            cumulantT4[k,j,i,l] = a
            cumulantT4[l,j,i,k] = a
            cumulantT4[k,l,i,j] = a
            cumulantT4[k,j,l,i] = a

        end
    end
    return Array(cumulantT4)
end
