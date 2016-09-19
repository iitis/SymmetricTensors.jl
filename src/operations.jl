#some advanced opperations not used for cumulant calculation
trace{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}) = mapreduce(i -> trace(bsdata.frame[i,i].value), +, 1:size(bsdata)[2])
vec{T <: AbstractFloat}(bsdata::BoxStructure{T}) = Base.vec(convert(Array, bsdata))
vecnorm{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}) = norm(vec(bsdata))

segmentmult{T <: AbstractFloat}(k1::Int, k2::Int, bsdata::BoxStructure{T, 2}) =
mapreduce(i -> readsegments([k1,i], bsdata)*readsegments([i,k2], bsdata), +, 1:size(bsdata.frame, 1))
segmentmult{T <: AbstractFloat}(k1::Int, k2::Int, bsdata::BoxStructure{T, 2}, bsdata1::BoxStructure{T, 2}) =
mapreduce(i -> readsegments([k1,i], bsdata)*readsegments([i,k2], bsdata1), +, 1:size(bsdata.frame, 1))

function generateperm(i::Int, ar::Array{Int})
    ret = ar
    ret[i], ret[1] = ar[1], ar[i]
    ret
end


function square{T <: AbstractFloat}(bsdata::BoxStructure{T, 2})
    s = size(bsdata)
    ret = NullableArray(Matrix{T}, size(bsdata.frame))
    for i = 1:s[2], j = i:s[2]
        @inbounds ret[i,j] = segmentmult(i,j, bsdata)
    end
    BoxStructure(ret)
end

function *{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}, bsdata1::BoxStructure{T, 2})
    s = size(bsdata)
    s == size(bsdata1) || throw(DimensionMismatch("dims of B1 $(size(bsdata)) must equal to dims of B2 $(size(bsdata1))"))
    ret = zeros(T, s[3], s[3])
    for i = 1:s[2], j = 1:s[2]
        @inbounds ret[seg(i, s[1], s[3]), seg(j, s[1], s[3])] = segmentmult(i,j, bsdata, bsdata1)
    end
    ret
end

function segmentmult{T <: AbstractFloat}(i::Int, j::Int, bsdata::BoxStructure{T, 2}, m::Array{T, 2})
  s = size(bsdata)
  mapreduce(k -> readsegments([i,k], bsdata)*(m[seg(k, s[1], size(m ,1)),seg(j, s[1], size(m ,2))]), +, 1:s[2])
end

function segmentmult{T <: AbstractFloat, N}(k::Array{Int, 1}, bsdata::BoxStructure{T, N}, m::Array{T, 2}, mode::Int = 1)
  s = size(bsdata)
  mapreduce(i -> Tensors.modemult(readsegments([i, k[2:end]...], bsdata),
  m[seg(k[1], s[1], size(m ,1)),seg(i, s[1], size(m ,2))], mode), +, 1:s[2])
end

function *{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}, mat::Matrix{T})
    s = size(bsdata)
    s[3] == size(mat,1) || throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(mat,1))"))
    ret = similar(mat)
    for i = 1:s[2], j = 1:ceil(Int, size(mat, 2)/s[1])
        @inbounds ret[seg(i, s[1], size(mat,1)), seg(j, s[1], size(mat,2))] = segmentmult(i,j, bsdata, mat)
    end
    ret
end

function modemult{T <: AbstractFloat, N}(bsdata::BoxStructure{T, N}, mat::Matrix{T}, mode::Int)
    s = size(bsdata)
    s[3] == size(mat,2) || throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(mat,1))"))
    ret = zeros(T, size(mat,1), fill(s[3], N-1)...)
    matseg = ceil(Int, size(mat, 1)/s[1])
    for i = 1:(s[2]^(N-1)*matseg)
        readind = ind2sub((matseg, fill(s[2], N-1)...), i)
        writeind = (map(k -> seg(readind[k], s[1], size(ret,k)), 1:N)...)
        @inbounds ret[writeind...] = segmentmult([readind...], bsdata, mat)
    end
    permutedims(ret, generateperm(mode, collect(1:N)))
end
#covariance

function covbs{T <: AbstractFloat}(data::Matrix{T}, segments::Int = 2, corrected::Bool = false)
    len = size(data,2)
    segsizetest(len, segments)
    (len%segments == 0)? () : segments += 1
    ret = NullableArray(Matrix{T}, segments, segments)
    segsize = ceil(Int, len/segments)
    for i = 1:segments, j = i:segments
        @inbounds ret[i,j] = cov(data[:,seg(i, segsize, len)], data[:,seg(j, segsize, len)], corrected = corrected)
    end
    BoxStructure(ret)
end



#below are bssc functions based on the kolda notation
#they may be used for some sort of ALS of boxes
function segmentmult{T <: AbstractFloat}(k::Int, m::Array{T, 2}, v::Array{Array{T, 2}})
  s = size(v[1], 1)
  mapreduce(i -> transpose(m[seg(i, s, size(m ,1)), seg(k, s, size(m ,2))])*(v[i]), +, 1:size(v , 1))
end

function bcss{T <: AbstractFloat}(bsdata::BoxStructure{T, 2}, m::Matrix{T})
    s = size(bsdata)
    s[3]  == size(m,1)||throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(m,1))"))
    segments = ceil(Int, size(m, 2)/s[1])
    ret = NullableArray(Array{T, 2}, segments, segments)
    for i = 1:segments
      temp = Array(Array{T, 2}, s[2])
      for k = 1:s[2]
          @inbounds temp[k] = segmentmult(k,i, bsdata, m)
      end
      for j = 1:i
	       @inbounds ret[j,i] = segmentmult(j, m, temp)
      end
   end
   BoxStructure(ret)
end

function bcsscel{T <: AbstractFloat, N}(bsdata::BoxStructure{T, N}, v::Array{T}...)
    ret = modemult(bsdata, v[1], N)
    for j = 2:N
        @inbounds ret = Tensors.modemult(ret, v[j], N-j+1)
    end
    ret[1]
end

function bcsseg{T <: AbstractFloat, N}(bsdata::BoxStructure{T, N}, r::Array{T, 2}...)
    dims = [map(i -> size(r[i], 1), 1:N)...]
    ret = zeros(T, dims...)
    for i = 1:mapreduce(k -> dims[k], *, 1:N)
        ind = ind2sub((dims...), i)
        @inbounds ret[ind...] = bcsscel(bsdata, map(k -> r[k][ind[k],:], 1:N)...)
    end
    ret
end

function bcssclass{T <: AbstractFloat, N}(bsdata::BoxStructure{T, N}, m::Matrix{T}, segments::Int = 2)
    len = size(m,1)
    segsizetest(len, segments)
    (len%segments == 0)? () : segments += 1
    ret = NullableArray(Array{T, N}, fill(segments, N)...)
    segsize = ceil(Int, len/segments)
    ind = indices(N, segments)
    for i in ind
        @inbounds ret[i...] = bcsseg(bsdata, map(k -> m[seg(i[k], segsize, len),:], 1:N)...)
    end
    BoxStructure(ret)
end



# those below are irrelavent
redundantcol(n::Int, s::Int) =  n*ceil(Int, s/n) - s

function convertlim{T<:AbstractFloat, N}(::Type{Array}, bsdata::BoxStructure{T,N}, limit::Int = 0)
  s = size(bsdata)
  (0 < limit < s[3])? (si = limit): (si = s[3])
  ret = zeros(T, fill(si, N)...)
    for i = 1:(s[2]^N)
        readind = ind2sub((fill(s[2], N)...), i)
        writeind = (map(k -> seg(readind[k], s[1], si), 1:N)...)
        range = map(k -> ((s[1]*readind[k] > si)? (1:(si%s[1])) : (1:s[1])), 1:N)
        ret[writeind...] = readsegments(collect(readind), bsdata)[range...]
      end
  ret
end


"""elementwise opertation that changes the value of the bs (the f!() function )

input bs and number (Real)

Returns single bs of the size of input bs
"""
function operation!{T<: AbstractFloat,N, S <: Real}(bsdata::BoxStructure{T,N}, op::Function, n::S)
      ind = indices(N, size(bsdata.frame, 1))
      for i in ind
        @inbounds bsdata.frame[i...] = op(bsdata.frame[i...].value, n)
      end
end

"""add function that changes the input data f!() type

input bs data to which a number is added elementwisely
"""
add{T <: AbstractFloat, S <: Real}(bsdata::BoxStructure{T}, n::S)  = operation!(bsdata, +, n)
