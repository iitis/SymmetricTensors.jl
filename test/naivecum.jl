
cumulantelement{T<:AbstractFloat}(d::Vector{T}...) = moment(d...) + mixedelements(d...)
moment{T<:AbstractFloat}(d::Vector{T}...) = mean(mapreduce(i -> d[i], .*, 1:length(d)))

function mixedelements{T<:AbstractFloat}(A::Vector{T},B::Vector{T},C::Vector{T},D::Vector{T})
  -mean(A.*B)*mean(C.*D) -mean(A.*C)*mean(B.*D) -mean(A.*D)*mean(B.*C)
end

function mixedelements{T<:AbstractFloat}(A::Vector{T},B::Vector{T},C::Vector{T},D::Vector{T},E::Vector{T})
  a = -mean(A.*B.*C)*mean(D.*E) - mean(A.*B.*D)*mean(C.*E) - mean(A.*B.*E)*mean(D.*C)
  a -= mean(D.*B.*C)*mean(A.*E)+ mean(E.*B.*C)*mean(D.*A) +mean(A.*D.*C)*mean(B.*E)
  a -= mean(A.*E.*C)*mean(B.*D)+ mean(D.*E.*C)*mean(A.*B)+ mean(D.*B.*E)*mean(A.*C)+mean(A.*D.*E)*mean(C.*B)
  return a
end


function naivecumulant{T<:AbstractFloat}(data::Matrix{T}, order::Int = 4)
    data = center(data)
    n = size(data, 2)
    ret = zeros(T, fill(n, order)...)
    if order in [2,3]
      @inbounds for i = 1:(n^order)
          ind = ind2sub((fill(n, order)...), i)
          ret[ind...] = moment(map(i -> data[:,ind[i]],1:order)...)
        end
    elseif order in [4,5]
      @inbounds for i = 1:(n^order)
          ind = ind2sub((fill(n, order)...), i)
          ret[ind...] = cumulantelement(map(i -> data[:,ind[i]],1:order)...)
        end
    end
    return ret
end
