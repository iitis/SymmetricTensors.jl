# uses the naive method to calculate cumulants 2 - 6

"""
the element of the nth cumulant tensor: a sum of the moment tensor's element
and mixed element

input n vectors of data

output float number
"""
cumulantelement{T<:AbstractFloat}(d::Vector{T}...) = moment(d...) + mixedelements(d...)


"""
the element of nth moment tensor
"""
moment{T<:AbstractFloat}(d::Vector{T}...) = mean(mapreduce(i -> d[i], .*, 1:length(d)))


"""
the mixed element for cumulants 4-6 respectivelly
"""
function mixedelements{T<:AbstractFloat}(A::Vector{T},B::Vector{T},C::Vector{T},D::Vector{T})
  -mean(A.*B)*mean(C.*D) -mean(A.*C)*mean(B.*D) -mean(A.*D)*mean(B.*C)
end
function mixedelements{T<:AbstractFloat}(A::Vector{T},B::Vector{T},C::Vector{T},D::Vector{T},E::Vector{T})
  a = -mean(A.*B.*C)*mean(D.*E) - mean(A.*B.*D)*mean(C.*E) - mean(A.*B.*E)*mean(D.*C)
  a -= mean(D.*B.*C)*mean(A.*E)+ mean(E.*B.*C)*mean(D.*A) +mean(A.*D.*C)*mean(B.*E)
  a -= mean(A.*E.*C)*mean(B.*D)+ mean(D.*E.*C)*mean(A.*B)+ mean(D.*B.*E)*mean(A.*C)+mean(A.*D.*E)*mean(C.*B)
  return a
end
function mixedelements{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T}, E::Vector{T}, F::Vector{T})
  block1 = -mean(A.*B.*C)*mean(D.*E.*F) - mean(A.*B.*D)*mean(C.*E.*F) - mean(A.*B.*E)*mean(C.*D.*F)
  block1 -= mean(A.*B.*F)*mean(C.*D.*E) + mean(A.*C.*D)*mean(B.*E.*F) + mean(A.*C.*E)*mean(B.*D.*F)
  block1 -= mean(A.*C.*F)*mean(B.*D.*E) + mean(A.*D.*E)*mean(B.*C.*F) + mean(A.*D.*F)*mean(B.*C.*E)
  block1 -= mean(A.*E.*F)*mean(B.*C.*D)
  block2 = -mean(A.*B.*C.*D)*mean(E.*F) - mean(A.*B.*C.*E)*mean(D.*F) - mean(A.*B.*C.*F)*mean(D.*E)
  block2 -= mean(A.*B.*D.*E)*mean(C.*F) + mean(A.*B.*D.*F)*mean(C.*E) + mean(A.*B.*E.*F)*mean(C.*D)
  block2 -= mean(A.*B)*mean(C.*D.*E.*F) + mean(A.*C.*D.*E)*mean(B.*F) + mean(A.*C.*D.*F)*mean(B.*E)
  block2 -= mean(A.*C.*E.*F)*mean(B.*D) + mean(A.*C)*mean(B.*D.*E.*F) + mean(A.*D.*E.*F)*mean(B.*C)
  block2 -= mean(A.*D)*mean(B.*C.*E.*F) + mean(A.*E)*mean(B.*C.*D.*F) + mean(A.*F)*mean(B.*C.*D.*E)
  block3 = -mean(A.*B)*mean(C.*D)*mean(E.*F) - mean(A.*B)*mean(C.*E)*mean(D.*F)
  block3 -= mean(A.*B)*mean(C.*F)*mean(D.*E) + mean(A.*C)*mean(B.*D)*mean(E.*F)
  block3 -= mean(A.*C)*mean(B.*E)*mean(D.*F) + mean(A.*C)*mean(B.*F)*mean(D.*E)
  block3 -= mean(A.*D)*mean(B.*C)*mean(E.*F) + mean(A.*E)*mean(B.*C)*mean(D.*F)
  block3 -= mean(A.*F)*mean(B.*C)*mean(D.*E) + mean(A.*D)*mean(B.*E)*mean(C.*F)
  block3 -= mean(A.*D)*mean(B.*F)*mean(C.*E) + mean(A.*E)*mean(B.*D)*mean(C.*F)
  block3 -= mean(A.*F)*mean(B.*D)*mean(C.*E) + mean(A.*E)*mean(B.*F)*mean(C.*D)
  block3 -= mean(A.*F)*mean(B.*E)*mean(C.*D)
  return block2+block1-2*block3
end


"""
calculates cumulant of given order

imput: data, and order

output: array{n}
"""
function naivecumulant{T<:AbstractFloat}(data::Matrix{T}, order::Int = 4)
    data = center(data)
    n = size(data, 2)
    ret = zeros(T, fill(n, order)...)
    if order in [2,3]
      @inbounds for i = 1:(n^order)
          ind = ind2sub((fill(n, order)...), i)
          ret[ind...] = moment(map(i -> data[:,ind[i]],1:order)...)
        end
    elseif order in [4,5,6]
      @inbounds for i = 1:(n^order)
          ind = ind2sub((fill(n, order)...), i)
          ret[ind...] = cumulantelement(map(i -> data[:,ind[i]],1:order)...)
        end
    end
    return ret
end
