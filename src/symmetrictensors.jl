"""Type constructor"""
immutable SymmetricTensor{T <: AbstractFloat, S}
    frame::NullableArrays.NullableArray{Array{T,S},S}
    sizesegment::Int
    function (::Type{SymmetricTensor}){T, S}(frame::NullableArrays.NullableArray{Array{T,S},S})
        structfeatures(frame)
        new{T, S}(frame, size(frame[fill(1,S)...].value,1))
    end
end

"""Unfold function.

Input: A - tensor, n - mode of unfold.

Returns: matrix.
"""
function unfold(A::Array, n::Int)
    C = setdiff(1:ndims(A), n)
    I = size(A)
    J = I[n]
    K = prod(I[C])
    return reshape(permutedims(A,[n; C]), J, K)
end

"""Tests if the array is symmetric for given tolerance.

Input: array, atol - tolerance

Returns: Assertion Error if failed
"""
function issymetric{T <: AbstractFloat, N}(array::Array{T, N}, atol::Float64 = 1e-7)
  for i=2:ndims(array)
     (maximum(abs(unfold(array, 1)-unfold(array, i))) < atol) ||
     throw(AssertionError("array not symmetric"))
  end
end

""" Tests the block size.

Input: n - size of data, s - size of block.

Returns: DimensionMismatch in failed.
"""
sizetest(n::Int, s::Int) = (n >= s > 0) || throw(DimensionMismatch("wrong segment size $s"))

"""Generates the tuple of sorted indices.

Input: N - size of the tuple, n - maximal index value.

Return Ordered tuple of indices (a multi-index)
"""
function indices(N::Int, n::Int)
    ret = Tuple{fill(Int, N)...}[]
    @eval begin
        @nloops $N i x -> (x==$N)? (1:$n): (i_{x+1}:$n) begin
            ind = @ntuple $N x -> i_{$N-x+1}
            @inbounds push!($ret, ind)
        end
    end
    ret
end

"""examine if data can be stored in the bs form....

search for expected exception
"""
function structfeatures{T <: AbstractFloat, N}(frame::NullableArray{Array{T,N},N})
  fsize = size(frame, 1)
  all(collect(size(frame)) .== fsize) ||
  throw(AssertionError("frame not square"))
  not_nulls = !frame.isnull
  !any(map(x->!issorted(ind2sub(not_nulls, x)), find(not_nulls))) ||
  throw(AssertionError("underdiagonal block not null"))
  for i in indices(N, fsize-1)
    @inbounds all(collect(size(frame[i...].value)) .== size(frame[i...].value, 1)) ||
    throw(AssertionError("[$i ] block not square"))
  end
  for i=1:fsize
    @inbounds issymetric(frame[fill(i, N)...].value)
  end
end

"""produces  set of indices for data in multidiemntional array
to read them in segments to perform bs

Return range
"""
seg(i::Int, of::Int, limit::Int) =  (i-1)*of+1 : ((i*of <= limit) ? i*of : limit)

"""Converts super-symmetric array into blocks.

Input: data - Array{N}, s - size of block.

Returns: Array{N} of blocks.
"""

function convert{T <: AbstractFloat, N}(::Type{SymmetricTensor}, data::Array{T, N}, s::Int = 2)
  issymetric(data)
  n = size(data,1)
  sizetest(n, s)
  q = ceil(Int, n/s)
  ret = NullableArray(Array{T, N}, fill(q, N)...)
  for writeind in indices(N, q)
      readind = map(k::Int -> seg(k, s, n), writeind)
      @inbounds ret[writeind...] = data[readind...]
  end
  SymmetricTensor(ret)
end

""" reads a segemnt with given multiindex,
if multiindex not sorted, find segment with sorted once and performs
  required permutation od fims

returns N dimentional Array
"""
function readsegments(i::Tuple, bt::SymmetricTensor)
  sortidx = sortperm([i...])
  permutedims(bt.frame[i[sortidx]...].value, invperm(sortidx))
end

"""gives the  number of boxes and the size of data stored in bs

input bs

Return Tuple of Ints
"""
function size{T <: AbstractFloat, N}(bt::SymmetricTensor{T, N})
  segsize = bt.sizesegment
  numseg = size(bt.frame, 1)
  numdata = segsize * (numseg-1) + size(bt.frame[end].value, 1)
  segsize, numseg, numdata
end

"""tests if sizes on many bs are the same - for elementwise opertation on may bs
"""
function testsize{T <: AbstractFloat, N}(bt::SymmetricTensor{T, N}...)
  for i = 2:size(bt,1)
    @inbounds size(bt[1]) == size(bt[i]) || throw(DimensionMismatch("dims of B1 $(size(bt[1])) must equal to dims of B$i $(size(bt[i]))"))
  end
end

""" converts bs into Array
"""
function convert{T<:AbstractFloat, N}(::Type{Array}, bt::SymmetricTensor{T,N})
  s = size(bt)
  ret = zeros(T, fill(s[3], N)...)
    for i = 1:(s[2]^N)
        readind = ind2sub((fill(s[2], N)...), i)
        writeind = map(k -> seg(readind[k], s[1], s[3]), 1:N)
        @inbounds ret[writeind...] = readsegments(readind, bt)
      end
  ret
end

convert{T<:AbstractFloat, N}(bt::SymmetricTensor{T,N}) = convert(Array, bt::SymmetricTensor{T,N})

convert{T<:AbstractFloat}(A::Vector{SymmetricTensor{T}}) = [convert(Array, A[i]) for i in 1:length(A)]

"""elementwise opertation on many bs

input many bs of the same size

Returns single bs of the size of input bs
"""
function operation{T<: AbstractFloat, N}(op::Function, A::SymmetricTensor{T,N}...)
  n = size(A, 1)
  (n > 1)? testsize(A...):()
  ret = similar(A[1].frame)
  for i in indices(N, size(A[1])[2])
    @inbounds ret[i...] = op(map(k ->  A[k].frame[i...].value, 1:n)...)
  end
  SymmetricTensor(ret)
end

"""elementwise opertation on bs and number

input bs and number (Real)

Returns single bs of the size of input bs
"""
function operation{T<: AbstractFloat, N}(op::Function, bt::SymmetricTensor{T,N}, num::Real)
  ret = similar(bt.frame)
  for i in indices(N, size(bt)[2])
    @inbounds ret[i...] = op(bt.frame[i...].value, num)
  end
  SymmetricTensor(ret)
end

operation(op::Function, a::Real, bt::SymmetricTensor) = operation(op, bt, a)

# implements simple operations on bs structure


for op = (:+, :-, :.*, :./)
  @eval ($op){T <: AbstractFloat, N}(bt::SymmetricTensor{T, N}, bt1::SymmetricTensor{T, N}) = operation($op, bt, bt1)
end


for op = (:+, :-, :*, :/)
  @eval ($op){T <: AbstractFloat, S <: Real}(bt::SymmetricTensor{T}, n::S)  = operation($op, bt, n)
end

for op = (:+, :*)
  @eval ($op){T <: AbstractFloat, S <: Real}(n::S, bt::SymmetricTensor{T})  = operation($op, bt, n)
end
