immutable SymmetricTensor{T <: AbstractFloat, S}
    frame::NullableArrays.NullableArray{Array{T,S},S}
    sizesegment::Int
    function call{T, S}(::Type{SymmetricTensor}, frame::NullableArrays.NullableArray{Array{T,S},S})
        structfeatures(frame)
        new{T, S}(frame, size(frame[fill(1,S)...].value,1))
    end
end


""" tests if the array is symmetric for a given tolerance

input array, tolerance
"""
function issymetric{T <: AbstractFloat, N}(data::Array{T, N}, atol::Float64 = 1e-7)
  for i=2:ndims(data)
     (maximum(abs(unfold(data, 1)-unfold(data, i))) < atol) || throw(AssertionError("array is not symmetric"))
  end
end

""" test wether the last segment of bs is not larger that an ordinary segment
"""
segsizetest(len::Int, segments::Int) = ((len%segments) <= (len÷segments)) || throw(DimensionMismatch("last segment len $(len-segments*(len÷segments)) > segment len $(len÷segments)"))

"""generates the set of sorted indices to run any operation on bs in a single loop.

input N - number of dimentions, n - maximal index value

Return Array of indices (ints)
todo moze da sie zrobic w tuplach
"""
function indices(N::Int, n::Int)
    ret = Vector{Int}[]
    @eval begin
        @nloops $N i x -> (x==$N)? (1:$n): (i_{x+1}:$n) begin
            ind = @ntuple $N x -> i_{$N-x+1}
            @inbounds push!($ret, [ind...])
        end
    end
    ret
end

"""examine if data can be stored in the bs form....

search for expected exception
"""
function structfeatures{T <: AbstractFloat, S}(frame::NullableArrays.NullableArray{Array{T,S},S})
  fsize = size(frame, 1)
  all(collect(size(frame)) .== fsize) || throw(AssertionError("frame not square"))
  not_nulls = !frame.isnull
  !any(map(x->!issorted(ind2sub(not_nulls, x)), find(not_nulls))) || throw(AssertionError("underdiagonal block not null"))
  for i in indices(S, fsize-1)
    @inbounds all(collect(size(frame[i...].value)) .== size(frame[i...].value, 1)) || throw(AssertionError("[$i ] block not square"))
  end
  for i=1:fsize
    @inbounds issymetric(frame[fill(i, S)...].value)
  end
end


"""produces  set of indices for data in multidiemntional array
to read them in segments to perform bs

Return range
"""
seg(i::Int, of::Int, limit::Int) =  (i-1)*of+1 : ((i*of <= limit) ? i*of : limit)


"""converts super-symmetric array into bs

input N dimentional Array

Returns N dimentional bs
"""
function convert{T <: AbstractFloat, N}(::Type{SymmetricTensor}, data::Array{T, N}, segments::Int = 2)
  issymetric(data)
  len = size(data,1)
  segsizetest(len, segments)
  (len%segments == 0)? () : segments += 1
  ret = NullableArray(Array{T, N}, fill(segments, N)...)
    ind = indices(N, segments)
    for writeind in ind
        readind = map(k::Int -> seg(k, ceil(Int, len/segments), len), writeind)
        @inbounds ret[writeind...] = data[readind...]
    end
  SymmetricTensor(ret)
end

""" reads a segemnt with given multiindex,
if multiindex not sorted, find segment with sorted once and performs
  required permutation od fims

returns N dimentional Array
"""
function readsegments(i::Vector{Int}, bs::SymmetricTensor)
  sortidx = sortperm(i)
  permutedims(bs.frame[i[sortidx]...].value, invperm(sortidx))
end

"""gives the  number of boxes and the size of data stored in bs

input bs

Return Tuple of Ints
"""
function size{T <: AbstractFloat, N}(bsdata::SymmetricTensor{T, N})
  segsize = bsdata.sizesegment
  numsegments = size(bsdata.frame, 1)
  numdata = segsize * (numsegments-1) + size(bsdata.frame[end].value, 1)
  segsize, numsegments, numdata
end

"""tests if sizes on many bs are the same - for elementwise opertation on may bs
"""
function testsize{T <: AbstractFloat, N}(bsdata::SymmetricTensor{T, N}...)
  for i = 2:size(bsdata,1)
    @inbounds size(bsdata[1]) == size(bsdata[i]) || throw(DimensionMismatch("dims of B1 $(size(bsdata[1])) must equal to dims of B$i $(size(bsdata[i]))"))
  end
end

""" converts bs into Array
"""
function convert{T<:AbstractFloat, N}(::Type{Array}, bsdata::SymmetricTensor{T,N})
  s = size(bsdata)
  ret = zeros(T, fill(s[3], N)...)
    for i = 1:(s[2]^N)
        readind = ind2sub((fill(s[2], N)...), i)
        writeind = map(k -> seg(readind[k], s[1], s[3]), 1:N)
        @inbounds ret[writeind...] = readsegments(collect(readind), bsdata)
      end
  ret
end

convert{T<:AbstractFloat, N}(bsdata::SymmetricTensor{T,N}) = convert(Array, bsdata::SymmetricTensor{T,N})

convert{T<:AbstractFloat}(c::Vector{SymmetricTensor{T}}) = [convert(Array, c[i]) for i in 1:length(c)]

"""elementwise opertation on many bs

input many bs of the same size

Returns single bs of the size of input bs
"""
function operation{T<: AbstractFloat, N}(op::Function, bsdata::SymmetricTensor{T,N}...)
  n = size(bsdata, 1)
  (n > 1)? testsize(bsdata...):()
  ret = similar(bsdata[1].frame)
  ind = indices(N, size(bsdata[1].frame, 1))
  for i in ind
    @inbounds ret[i...] = op(map(k ->  bsdata[k].frame[i...].value, 1:n)...)
  end
  SymmetricTensor(ret)
end

"""elementwise opertation on bs and number

input bs and number (Real)

Returns single bs of the size of input bs
"""
function operation{T<: AbstractFloat, N}(op::Function, bsdata::SymmetricTensor{T,N}, a::Real)
  ret = similar(bsdata.frame)
  ind = indices(N, size(bsdata.frame, 1))
  for i in ind
    @inbounds ret[i...] = op(bsdata.frame[i...].value, a)
  end
  SymmetricTensor(ret)
end

operation(op::Function, a::Real, bsdata::SymmetricTensor) = operation(op, bsdata, a)

# implements simple operations on bs structure


for op = (:+, :-, :.*, :./)
  @eval ($op){T <: AbstractFloat, N}(bsdata::SymmetricTensor{T, N}, bsdata1::SymmetricTensor{T, N}) = operation($op, bsdata, bsdata1)
end


for op = (:+, :-, :*, :/)
  @eval ($op){T <: AbstractFloat, S <: Real}(bsdata::SymmetricTensor{T}, n::S)  = operation($op, bsdata, n)
end

for op = (:+, :*)
  @eval ($op){T <: AbstractFloat, S <: Real}(n::S, bsdata::SymmetricTensor{T})  = operation($op, bsdata, n)
end
