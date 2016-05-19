module SymmetricMatrix
using NullableArrays
using Iterators
using Tensors
import Base: trace, vec, vecnorm, +, -, *, .*, /, ./, size, transpose

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
        (maximum(abs(unfold(data, 1)-unfold(data, i))) < 1e-7) || throw(DimensionMismatch("array is not symmetric"))
    end
end
function segsizetest{T <: AbstractFloat}(data::Array{T}, segments::Int)
    ((size(data,1)%segments) <= (size(data,1)÷segments)) || throw(DimensionMismatch("last segment len $(size(data,1)-segments*(size(data,1)÷segments)) > segment len $(size(data,1)÷segments)"))
end

issquared{T <: AbstractFloat, S}(ar::(Union{NullableArrays.NullableArray{Array{T,S},S}, Array{T,S}})) = (maximum(size(ar)) == minimum(size(ar)) )

function segmentise{T <: AbstractFloat}(data::Array{T}, segments::Int)
    issymetric(data)
    segsizetest(data, segments)
    step = [size(data,1)÷segments, size(data, 1)]
    creatnarray(data, (data::Array{T}, i::Int...) -> data[map(k::Int -> seg(k, step...), i)...], segments)
end

function structfeatures{T <: AbstractFloat, S}(frame::NullableArrays.NullableArray{Array{T,S},S})
     dims = ndims(frame)
     issquared(frame) || throw(DimensionMismatch("frame not squared"))
     for i in product(fill(1:size(frame, 1), dims)...)
	if !issorted(i)
	  isnull(frame[i...]) || throw(ArgumentError("underdiagonal block [$i ] not null"))
	elseif maximum(i) < size(frame, 1)
	  issquared(frame[i...].value) || throw(DimensionMismatch("[$i ] block not squared"))  
	end
    end
    for k = 1:size(frame, 1)
        issymetric(frame[fill(k,dims)...].value)
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

transpose{T <: AbstractFloat}(bs::BoxStructure{T}, i::Array{Int}) = permutedims(bs.frame[sort(i)...].value, invperm(sortperm(i))) 

size{T <: AbstractFloat}(bsdata::BoxStructure{T}) = bsdata.sizesegment, size(bsdata.frame, 1), bsdata.sizesegment*(size(bsdata.frame, 1)-1)+size(bsdata.frame[end,end].value,1)
convert{T <: AbstractFloat}(::Type{BoxStructure{T}}, data::Array{T}, segments::Int = 2) = BoxStructure(segmentise(data, segments))

readsegments{T <: AbstractFloat}(i::Array{Int, 1}, data::BoxStructure{T}) = transpose(data, i)
 #     return isnull(data.frame[i...])? transpose(data.frame[sort(i)...].value): data.frame[i...].value


segmentmult{T <: AbstractFloat}(k::Array{Int, 1}, bsdata::BoxStructure{T}...) = mapreduce(i -> readsegments([k[1],i], bsdata[1])*readsegments([i,k[2]], bsdata[size(bsdata,1)]), +, 1:size(bsdata[1].frame, 1))

function testsize{T <: AbstractFloat}(bsdata::BoxStructure{T}...)
    for i = 2:size(bsdata,1)
        size(bsdata[1]) == size(bsdata[i]) || throw(DimensionMismatch("dims of B1 $(size(bsdata[1])) must equal to dims of B2 $(size(bsdata[i]))"))
    end
end

function bstoarrayf{T <: AbstractFloat}(bsfunction::Function, dims::Int, bsdata::BoxStructure{T}...)
    testsize(bsdata...)
    s = size(bsdata[1])
    ret = zeros(T, fill(s[3], dims)...)
    for k in product(fill(1:s[2], dims)...)
        ret[(map(i -> seg(k[i], s[1], s[3]), 1:dims))...] = bsfunction(collect(k), bsdata...)
    end
    ret
end
#  map((x)-> bsfunction(collect(x), bsdata...), product(1:limit, 1:limit))

function bstobsf{T <: AbstractFloat}(f::Function, bsdata::BoxStructure{T}...)
    dim = ndims(bsdata[1].frame)
    ret = NullableArray(Array{T, dim}, size(bsdata[1].frame))
      for i in product(fill(1:size(bsdata[1])[2], dim)...)
	  if issorted(i) 
	    ret[i...] = f([i...], bsdata...)::Array{T}
	  end
      end
      BoxStructure(ret)
end

function blockop{T <: AbstractFloat}(f::Function, bsdata::BoxStructure{T}...)
    testsize(bsdata...)
    bstobsf((i, bsdata...) -> f(map(k -> bsdata[k].frame[i...].value,1:size(bsdata,1))...)::Array{T} ,bsdata...)
end

blockop{T <: AbstractFloat, S <: Real}(f::Function, bsdata::BoxStructure{T}, n::S) = bstobsf((i, bsdata) -> f(bsdata.frame[i...].value, n)::Array{T} ,bsdata)
convert{T <: AbstractFloat}(::Type{Array{T}}, bsdata::BoxStructure{T}) = bstoarrayf(readsegments, ndims(bsdata.frame), bsdata)

#operations
for op = (:+, :-, :*, :/)
  @eval ($op){T <: AbstractFloat, S <: Real}(bsdata::BoxStructure{T}, n::S) = blockop($op, bsdata, n)
end
for op = (:+, :-, :.*, :./)
  @eval ($op){T <: AbstractFloat}(bsdata::BoxStructure{T}, bsdata1::BoxStructure{T}) = blockop($op, bsdata, bsdata1)
end
square{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}) = bstobsf(segmentmult, bsdata)
trace{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}) = mapreduce(i -> trace(bsdata.frame[i,i].value), +, 1:size(bsdata)[2])
vecnorm{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}) = sqrt(trace(square(bsdata)))
vec{T <: AbstractFloat}(bsdata::BoxStructure{T}) = Base.vec(convert(Array{Float64}, bsdata))

#multiplications 
*{T <: AbstractFloat}(bsdata::BoxStructure{T, 2},  bsdata1::BoxStructure{T, 2}) = bstoarrayf(segmentmult, 2, bsdata, bsdata1)
slisemat{T <: AbstractFloat}(mat::Matrix{T}, slisesize::Int) = map(i -> mat[:,seg(i, slisesize, size(mat, 2))], 1:ceil(Int, size(mat,2)/slisesize))
function *{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}, mat::Matrix{T})
    s = size(bsdata)
    s[3] == size(mat,1) || throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(mat,1))"))
    hcat(map(k -> vcat(map(k1 -> (mapreduce(j -> readsegments([k1,j], bsdata)*slisemat(mat, s[1])[k][seg(j, s[1], s[3]),:], +, 1:s[2])), 1:s[2])...), 1:ceil(Int, size(mat,2)/s[1]))...)
end

#covariance
function covbs{T <: AbstractFloat}(data::Matrix{T}, segments::Int = 2, corrected::Bool = false)
    segsizetest(transpose(data), segments)
    BoxStructure(creatnarray(data, (data::Matrix{T}, b1::Int, b2::Int) -> cov(data[:,seg(b1, size(data,2)÷segments, size(data, 2))], data[:,seg(b2, size(data,2)÷segments, size(data, 2))], corrected = corrected), segments))
end

export BoxStructure, convert, +, -, *, /, trace, vec, square, vecnorm, covbs
end

# dokladnosci przy dodawaniu


