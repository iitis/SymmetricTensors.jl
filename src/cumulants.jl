cumulant1{T<:AbstractFloat}(data::Matrix{T}) = mean(data,1)

cumulant2{T<:AbstractFloat}(data::Matrix{T}) = (cov(data, corrected = false))

function cumulant3{T<:AbstractFloat}(data::Matrix{T})
    centred = zeros(data)
    n = size(data, 2)
    for i = 1:n
        centred[:,i] = data[:,i]-mean(data[:,i])
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
        end
    end
    return Array(cumulantT3)
end

E{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}) = mean(A.*B.*C.*D)

E31{T<:AbstractFloat}(M1::Vector{T}, M2::Vector{T}, M3::Vector{T}, N::Vector{T}) = mean(M1.*M2.*M3)*mean(N)
p31{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}) = E31(A,B,C,D)+E31(A,B,D,C)+E31(A,D,C,B)+E31(D,B,C,A)

E22{T<:AbstractFloat}(M1::Vector{T}, M2::Vector{T}, N1::Vector{T}, N2::Vector{T}) = mean(M1.*M2)*mean(N1.*N2)
p22{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}) = E22(A,B,C,D)+E22(A,C,B,D)+E22(A,D,C,B)

E211{T<:AbstractFloat}(M1::Vector{T}, M2::Vector{T}, N::Vector{T}, O::Vector{T}) =
  mean(M1.*M2)*mean(N)*mean(O)

p211{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}) =
  E211(A,B,C,D)+E211(A,C,B,D)+E211(A,D,C,B)+E211(C,B,A,D)+E211(C,D,A,B)+E211(B,D,A,C)

mixedelements{T<:AbstractFloat}(A::Vector{T},B::Vector{T},C::Vector{T},D::Vector{T}) = -p31(A,B,C,D)-p22(A,B,C,D)+2*p211(A,B,C,D)
cumulant4element{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}) =
  (E(A,B,C,D)+mixedelements(A,B,C,D)-6*mean(A)*mean(B)*mean(C)*mean(D))

function cumulant4{T<:AbstractFloat}(data::Matrix{T})
    n = size(data, 2)
    cumulantT4 = SharedArray(T, n, n, n, n)

    @sync @parallel for i = 1:n
      for  j = i:n, k = j:n, l = k:n
            a = cumulant4element(data[:,i], data[:,j], data[:,k], data[:,l])
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
