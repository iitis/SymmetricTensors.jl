"""Type constructor

Instances: frame - stroes arrays on nullable arrays
bls - Int, size of ordinary block
bln - Int, number of blocks
datasize - Int, size od data stored (in each direction the same)
sqr - Bool, is the last block size a same sa ordinary block size
"""
immutable SymmetricTensor{T <: AbstractFloat, N}
    frame::NullableArray{Array{T,N},N}
    bls::Int
    bln::Int
    dats::Int
    sqr::Bool
    function (::Type{SymmetricTensor}){T, N}(frame::NullableArray{Array{T,N},N}, test::Bool = true)
        s = size(frame[fill(1,N)...].value,1)
        g = size(frame, 1)
        last_block = size(frame[end].value, 1)
        m = s * (g-1) + last_block
        if test
          frtest(frame, s, g)
        end
        new{T, N}(frame, s, g, m, s == last_block)
    end
end

"""Unfold function.

Input: A - tensor, n - mode of unfold.

Returns: matrix.
"""
function unfold(ar::Array, n::Int)
    C = setdiff(1:ndims(ar), n)
    i = size(ar)
    k = prod(i[C])
    return reshape(permutedims(ar,[n; C]), i[n], k)
end

"""Tests if the array is symmetric for given tolerance.

Input: array, atol - tolerance

Returns: Assertion Error if failed
"""
function issymetric{T <: AbstractFloat, N}(ar::Array{T, N}, atol::Float64 = 1e-7)
  for i=2:N
     maximum(abs(unfold(ar, 1)-unfold(ar, i))) < atol ||throw(AssertionError("not symmetric"))
  end
end

"""Examines if data can be stored in SymmetricTensor form.

Input: data - nullable array of arrays.

Returns: Assertion error if: all sizes of nullable array not equal, at least
  one undergiagonal block not null, at lest one (not last) block not squared,
   at lest one diagonal block not symmetric.
"""
function frtest{T <: AbstractFloat, N}(data::NullableArray{Array{T,N},N}, s::Int, g::Int)
  all(collect(size(data)) .== g) || throw(AssertionError("frame not square"))
  not_nulls = !data.isnull
  !any(map(x->!issorted(ind2sub(not_nulls, x)), find(not_nulls))) ||
  throw(AssertionError("underdiag. block not null"))
  for i in indices(N, g-1)
    @inbounds all(collect(size(data[i...].value)) .== s)|| throw(AssertionError("$i block not square"))
  end
  for i=1:g
    @inbounds issymetric(data[fill(i, N)...].value)
  end
end

"""Generates the tuple of sorted indices.

Input: N - size of the tuple, n - maximal index value.

Return Ordered tuple of indices (a multi-index)
"""
function indices(N::Int, m::Int)
    ret = Tuple{fill(Int, N)...}[]
    @eval begin
        @nloops $N i x -> (x==$N)? (1:$m): (i_{x+1}:$m) begin
            @inbounds ind = @ntuple $N x -> i_{$N-x+1}
            @inbounds push!($ret, ind)
        end
    end
    ret
end

""" Tests the block size.

Input: n - size of data, s - size of block.

Returns: DimensionMismatch in failed.
"""
sizetest(m::Int, s::Int) = (m >= s > 0)|| throw(DimensionMismatch("bad block size $s > $m"))

""" Helper, gives a value of Nullable arrays inside Symmetric Tensor, at given
tuple of multi indices
"""
val{T<: AbstractFloat, N}(st::SymmetricTensor{T,N}, i::Tuple) = st.frame[i...].value

"""Produces a range of indices to determine the block.

Input: i - block's number, s - block's size, n - data size.

Returns: range.
"""
seg(i::Int, s::Int, m::Int) = (i-1)*s+1 : ((i*s <= m)? i*s : m)

"""Converts super-symmetric array into blocks.

Input: data - Array{N}, s - size of block.

Returns: Array{N} of blocks.
"""

function convert{T <: AbstractFloat, N}(::Type{SymmetricTensor}, data::Array{T, N}, s::Int = 2)
  issymetric(data)
  m = size(data,1)
  sizetest(m, s)
  q = ceil(Int, m/s)
  ret = NullableArray(Array{T, N}, fill(q, N)...)
  for writeind in indices(N, q)
      @inbounds readind = map(k::Int -> seg(k, s, m), writeind)
      @inbounds ret[writeind...] = data[readind...]
  end
  SymmetricTensor(ret)
end

"""Reads a block, given multiindex If multiindex is not sorted,
sorts it, finds a block and permutes dimentions of output.

Imput: i - mulitiindex tuple, st: - SymmetricTensor

Returns: Array
"""
function readsegments(st::SymmetricTensor, i::Tuple)
  ind = sortperm([i...])
  permutedims(val(st, i[ind]), invperm(ind))
end

"""Gives features of SymmetricTensor object

input st - SymmetricTensor object

Return: block's size, number of blocks, data size
"""
size{T <: AbstractFloat, N}(st::SymmetricTensor{T, N}) = st.bls, st.bln, st.dats

"""Converts Symmetric Tensor object to Array
"""
function convert{T<:AbstractFloat, N}(::Type{Array}, st::SymmetricTensor{T,N})
  s, g , m = size(st)
  ret = zeros(T, fill(m, N)...)
    for i = 1:(g^N)
        @inbounds readind = ind2sub((fill(g, N)...), i)
        @inbounds writeind = map(k -> seg(readind[k], s, m), 1:N)
        @inbounds ret[writeind...] = readsegments(st, readind)
      end
  ret
end
convert{T<:AbstractFloat, N}(st::SymmetricTensor{T,N}) = convert(Array, st::SymmetricTensor{T,N})

"""Converts vector of Symmetric Tensors to vector of Arrays
"""
convert{T<:AbstractFloat}(c::Vector{SymmetricTensor{T}}) = [convert(Array, c[i]) for i in 1:length(c)]

# ---- basic operations on Symmetric Tensors ----

"""Tests if sizes on many  are the same
"""
function testsize{T <: AbstractFloat, N}(st::SymmetricTensor{T, N}...)
  for i = 2:size(st,1)
    @inbounds size(st[1]) == size(st[i]) ||
    throw(DimensionMismatch("dims of B1 $(size(bt[1])) must equal dims of B$i $(size(bt[i]))"))
  end
end

"""Elementwise opertation on many Symmetric Tensors objects

Input op - opperation funtction,

Returns single bs of the size of input bs
"""
function operation{T<: AbstractFloat, N}(op::Function, st::SymmetricTensor{T,N}...)
  r = size(st, 1)
  (r > 1)? testsize(st...):()
  ret = similar(st[1].frame)
  for i in indices(N, st[1].bln)
    @inbounds ret[i...] = op(map(k -> val(st[k], i), 1:r)...)
  end
  SymmetricTensor(ret)
end

"""elementwise opertation on bs and number

input bs and number (Real)

Returns single bs of the size of input bs
"""
function operation{T<: AbstractFloat, N}(op::Function, st::SymmetricTensor{T,N}, num::Real)
  ret = similar(st.frame)
  for i in indices(N, st.bln)
    @inbounds ret[i...] = op(val(st, i), num)
  end
  SymmetricTensor(ret)
end

operation(op::Function, a::Real, st::SymmetricTensor) = operation(op, st, a)

# implements simple operations on bs structure

for op = (:+, :-, :.*, :./)
  @eval ($op){T <: AbstractFloat, N}(st::SymmetricTensor{T, N}...) =
  operation($op, st...)
end

for op = (:+, :-, :*, :/)
  @eval ($op){T <: AbstractFloat, S <: Real}(st::SymmetricTensor{T}, n::S) =
  operation($op, st, n)
end

for op = (:+, :*)
  @eval ($op){T <: AbstractFloat, S <: Real}(n::S, st::SymmetricTensor{T}) =
  operation($op, st, n)
end
