module SymmetricMatrix
using NullableArrays
using Iterators
using Tensors
import Base: trace, vec, vecnorm, +, *, .*, size, transpose

seg(i::Int, of::Int, limit::Int) = (i*of <= limit)? (((i-1)*of+1):(i*of)): (((i-1)*of+1):limit)


function creatnarray{T <: AbstractFloat}(data::Array{T}, rule::Function, segments::Int)
    arraysize = (size(data,1)%segments == 0)? segments: segments+1
    ret = NullableArray(Array{T, ndims(data)}, fill(arraysize, ndims(data))...)
    for i in product(fill(1:arraysize, ndims(data))...)
      if issorted(i) 
        ret[i...] = rule(data, i...)::Array{T}
      end
    end
    ret
end

function issymetric{T <: AbstractFloat}(data::Array{T})
    for i = 2:ndims(data)
        (maximum(abs(unfold(data, 1)-unfold(data, i))) < 1e-7) || throw(DimensionMismatch("matrix is not symmetric"))
    end
end

function segsizetest{T <: AbstractFloat}(data::Array{T}, segments::Int)
    ((size(data,1)%segments) <= (size(data,1)÷segments)) || throw(DimensionMismatch("last segment len $(size(data,1)-segments*(size(data,1)÷segments)) > segment len $(size(data,1)÷segments)"))
end

function segmentise{T <: AbstractFloat}(data::Array{T}, segments::Int)
    issymetric(data)
    segsizetest(data, segments)
    step = [size(data,1)÷segments, size(data, 1)]
    creatnarray(data, (data::Array{T}, i::Int...) -> data[map(k::Int -> seg(k, step...), i)...], segments)
end


function structfeatures(frame)
    maximum(size(frame)) == minimum(size(frame)) || throw(DimensionMismatch("frame not symmetric")) #porownanie wielu
     for i in product(fill(1:size(frame, 1), ndims(frame))...)
	if !issorted(i)
	  isnull(frame[i...]) || throw(ArgumentError("underdiagonal block [$i ] not null"))
	elseif maximum(i) < size(frame, 1)
	  maximum(size(frame[i...].value)) == minimum(size(frame[i...].value)) || throw(DimensionMismatch("[$i ] block not squared"))     
	end
    end
    for k = 1:size(frame, 1)
        issymetric(frame[fill(k,ndims(frame))...].value)
    end
end

immutable BoxStructure{T <: AbstractFloat, S}
    frame::NullableArrays.NullableArray{Array{T,S},S}
    sizesegment::Int
    function call{T, S}(::Type{BoxStructure}, frame::NullableArrays.NullableArray{Array{T,S},S})
        structfeatures(frame)
        new{T, S}(frame, size(frame[fill(1,S)...].value,1))
    end
end


transpose{T <: AbstractFloat}(bs::BoxStructure{T}, i::Array{Int}) = permutedims(bs.frame[sort(i)...].value, invperm(sortperm(i))) #cos tu jest zle

size{T <: AbstractFloat}(m1::BoxStructure{T}) = m1.sizesegment, size(m1.frame, 1), m1.sizesegment*(size(m1.frame, 1)-1)+size(m1.frame[end,end].value,1)
convert{T <: AbstractFloat}(::Type{BoxStructure{T}}, data::Array{T}, segments::Int = 2) = BoxStructure(segmentise(data, segments))

readsegments{T <: AbstractFloat}(i::Array{Int, 1}, data::BoxStructure{T}) = transpose(data, i)
 #     return isnull(data.frame[i...])? transpose(data.frame[sort(i)...].value): data.frame[i...].value


segmentmult{T <: AbstractFloat}(k::Array{Int, 1}, m1::BoxStructure{T}...) = mapreduce(i -> readsegments([k[1],i], m1[1])*readsegments([i,k[2]], m1[size(m1,1)]), +, 1:size(m1[1].frame, 1))


function testsize{T <: AbstractFloat}(m1::BoxStructure{T}...)
    for i = 2:size(m1,1)
        size(m1[1]) == size(m1[i]) || throw(DimensionMismatch("dims of B1 $(size(m1[1])) must equal to dims of B2 $(size(m1[i]))"))
    end
end


function bsoperation1{T <: AbstractFloat}(bsfunction::Function, dims::Int,  m1::BoxStructure{T}...)
    testsize(m1...)
    limit = size(m1[1])[2]
  #  map((x)-> bsfunction(collect(x), m1...), product(1:limit, 1:limit))
    cat(2, map(j -> cat(1, map(i -> bsfunction([i,j], m1...)::Array{T}, 1:limit)...), 1:limit)...)
#    eval(c(2, limit, bsfunction))

end

function bsoperation{T <: AbstractFloat}(bsfunction::Function, dims::Int, m1::BoxStructure{T}...)
    testsize(m1...)
    s = size(m1[1])
    ret = zeros(T, fill(s[3], dims)...)
    for k in product(fill(1:s[2], dims)...)
        ret[(map(i -> seg(k[i], s[1], s[3]), 1:dims))...] = bsfunction(collect(k), m1...)
    end
    ret
end

function operationonbs{T <: AbstractFloat}(f::Function, m1::BoxStructure{T}...)
    dim = ndims(m1[1].frame)
    ret = NullableArray(Array{T, dim}, size(m1[1].frame))
      for i in product(fill(1:size(m1[1])[2], dim)...)
	  if issorted(i) 
	    ret[i...] = f([i...], m1...)::Array{T}
	  end
      end
      BoxStructure(ret)
end

function blockop{T <: AbstractFloat, S <: Real}(n::S, f::Function, m1::BoxStructure{T}...)
    testsize(m1...)
    operationonbs((i, m1...) -> f(n, map(k -> m1[k].frame[i...].value,1:size(m1,1))...)::Array{T} ,m1...)
end

matricise{T <: AbstractFloat}(m1::BoxStructure{T}) = bsoperation(readsegments, ndims(m1.frame), m1)
*{T <: AbstractFloat}(m1::BoxStructure{T},  m2::BoxStructure{T}) = bsoperation(segmentmult, 2, m1, m2)

*{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S) = blockop(n, *, m1)
+{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S) = blockop(n, +, m1)
+{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T}) = blockop(0, +, m1, m2)
.*{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T}) = blockop(1, (a::Int,b::Array{T},c::Array{T}) -> b.*c, m1, m2)
square{T <: AbstractFloat}(m1::BoxStructure{T}) = operationonbs(segmentmult, m1)
trace{T <: AbstractFloat}(m1::BoxStructure{T}) = mapreduce(i -> trace(m1.frame[i,i].value), +, 1:size(m1)[2])
vecnorm{T <: AbstractFloat}(m1::BoxStructure{T}) = sqrt(trace(square(m1)))
vec{T <: AbstractFloat}(m1::BoxStructure{T}) = Base.vec(matricise(m1))

slisemat{T <: AbstractFloat}(m2::Matrix{T}, slisesize::Int) = map(i -> m2[:,seg(i, slisesize, size(m2, 2))], 1:ceil(Int, size(m2,2)/slisesize))

function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::Matrix{T})
    s = size(m1)
    s[3] == size(m2,1) || throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(m2,1))"))
    hcat(map(k -> vcat(map(k1 -> (mapreduce(j -> readsegments([k1,j], m1)*slisemat(m2, s[1])[k][seg(j, s[1], s[3]),:], +, 1:s[2])), 1:s[2])...), 1:ceil(Int, size(m2,2)/s[1]))...)
end
 #bs times vector

function covbs{T <: AbstractFloat}(data::Matrix{T}, segments::Int = 2, corrected::Bool = false)
    ((size(data,2)%segments) <= (size(data,2)÷segments)) || throw(DimensionMismatch("last segment len $(size(data,2)-segments*(size(data,1)÷segments)) > segment len $(size(data,2)÷segments)"))
    BoxStructure(creatnarray(data, (data::Matrix{T}, b1::Int, b2::Int) -> cov(data[:,seg(b1, size(data,2)÷segments, size(data, 2))], data[:,seg(b2, size(data,2)÷segments, size(data, 2))], corrected = corrected), segments))
end

export BoxStructure, convert, matricise, trace, vec, *, square, vecnorm, +, covbs, segmentise, BoxStructure1, structfeatures, readsegments
end

# dokladnosci przy dodawaniu


