cumulant1{T<:AbstractFloat}(data::Matrix{T}) = mean(data,1)

cumulant2{T<:AbstractFloat}(data::Matrix{T}) = (cov(data, corrected = false))

function cumulant3{T<:AbstractFloat}(data::Matrix{T})
    centred = zeros(data)
    n = size(data, 2)
    for i = 1:n
        centred[:,i] = data[:,i]-mean(data[:,i])
    end
    cumulantT3 = zeros(n,n,n)
    for i = 1:n
        for j = i:n, k = j:n
            a = mean(centred[:,i].*centred[:,j].*centred[:,k])
            cumulantT3[i,j,k] = a
            cumulantT3[i,k,j] = a
            cumulantT3[j,i,k] = a
            cumulantT3[j,k,i] = a
            cumulantT3[k,i,j] = a
            cumulantT3[k,j,i] = a
        end
    end
    return Array(cumulantT3)
end


mixedelements{T<:AbstractFloat}(A::Vector{T},B::Vector{T},C::Vector{T},D::Vector{T}) = -mean(A.*B)*mean(C.*D) -mean(A.*C)*mean(B.*D) -mean(A.*D)*mean(B.*C)

cumulant4element{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}) = mean(A.*B.*C.*D) + mixedelements(A,B,C,D)

function cumulant4{T<:AbstractFloat}(data::Matrix{T})
    n = size(data, 2)
    
    centred = zeros(data)
    for i = 1:n
        centred[:,i] = data[:,i]-mean(data[:,i])
    end
    
    cumulantT4 = SharedArray(T, n, n, n, n)

    @sync @parallel for i = 1:n
      for  j = i:n, k = j:n, l = k:n
            a = cumulant4element(centred[:,i], centred[:,j], centred[:,k], centred[:,l])
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

norm_tensor(A) = (norm(vec(A), 2)^2)/2
