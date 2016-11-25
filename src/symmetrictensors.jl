"""Type constructor"""
immutable SymmetricTensor{T <: AbstractFloat, N}
    frame::NullableArray{Array{T,N},N}
    blocksize::Int
    function (::Type{SymmetricTensor}){T, N}(frame::NullableArray{Array{T,N},N})
        structfeatures(frame)
        new{T, N}(frame, size(frame[fill(1,N)...].value,1))
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

"""Examines if data can be stored in SymmetricTensor form.

Input: data - nullable array of arrays.

Returns: Assertion error if: all sizes of nullable array not equal, at least
  one undergiagonal block not null, at lest one (not last) block not squared,
   at lest one diagonal block not symmetric.
"""
function structfeatures{T <: AbstractFloat, N}(data::NullableArray{Array{T,N},N})
  fsize = size(data, 1)
  all(collect(size(data)) .== fsize) ||
  throw(AssertionError("frame not square"))
  not_nulls = !data.isnull
  !any(map(x->!issorted(ind2sub(not_nulls, x)), find(not_nulls))) ||
  throw(AssertionError("underdiag. block not null"))
  for i in indices(N, fsize-1)
    @inbounds all(collect(size(data[i...].value)) .== size(data[i...].value, 1)) ||
    throw(AssertionError("[$i ] block not square"))
  end
  for i=1:fsize
    @inbounds issymetric(data[fill(i, N)...].value)
  end
end

"""Produces a range of indices to determine the block.

Input: i - block's number, s - block's size, n - data size.

Returns: range.
"""
seg(i::Int, s::Int, n::Int) = (i-1)*s+1 : ((i*s <= n)? i*s : n)

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

"""Reads a block, given multiindex If multiindex is not sorted,
sorts it, finds a block and permutes dimentions of output.

Imput: i - mulitiindex tuple, st: - SymmetricTensor

Returns: Array
"""
function readsegments(i::Tuple, st::SymmetricTensor)
  ind = sortperm([i...])
  permutedims(st.frame[i[ind]...].value, invperm(ind))
end

"""Gives features of SymmetricTensor object

input st - SymmetricTensor object

Return: s - block's size, g - number of blocks, n - data size
"""
function size{T <: AbstractFloat, N}(st::SymmetricTensor{T, N})
  s = st.blocksize
  g = size(st.frame, 1)
  n = s * (g-1) + size(st.frame[end].value, 1)
  s, g, n
end

"""Converts Symmetric Tensor object to Array
"""
function convert{T<:AbstractFloat, N}(::Type{Array}, st::SymmetricTensor{T,N})
  s = size(st)
  ret = zeros(T, fill(s[3], N)...)
    for i = 1:(s[2]^N)
        readind = ind2sub((fill(s[2], N)...), i)
        writeind = map(k -> seg(readind[k], s[1], s[3]), 1:N)
        @inbounds ret[writeind...] = readsegments(readind, st)
      end
  ret
end
convert{T<:AbstractFloat, N}(st::SymmetricTensor{T,N}) = convert(Array, st::SymmetricTensor{T,N})

"""Converts vector of Symmetric Tensors to vector of Arrays
"""
convert{T<:AbstractFloat}(A::Vector{SymmetricTensor{T}}) = [convert(Array, A[i]) for i in 1:length(A)]

# ---- basic operations on Symmetric Tensors ----

"""Tests if sizes on many bs are the same - for elementwise opertation on may bs
"""
function testsize{T <: AbstractFloat, N}(bt::SymmetricTensor{T, N}...)
  for i = 2:size(bt,1)
    @inbounds size(bt[1]) == size(bt[i]) ||
    throw(DimensionMismatch("dims of B1 $(size(bt[1])) must equal dims of B$i $(size(bt[i]))"))
  end
end

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
