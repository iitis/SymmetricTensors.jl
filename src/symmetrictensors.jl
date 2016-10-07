
"""unfold function from Tensors Gawron"""
function unfold(A::Array, n::Int)
    C = setdiff(1:ndims(A), n)
    I = size(A)
    J = I[n]
    K = prod(I[C])
    return reshape(permutedims(A,[n; C]), J, K)
end


""" tests if the array is symmetric for a given tolerance

input array, tolerance
"""
function issymetric{T <: AbstractFloat, N}(data::Array{T, N}, atol::Float64 = 1e-7)
  length(data)>0? () : return
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
  g = ceil(Int, len/segments)
  for writeind in indices(N, segments)
      readind = map(k::Int -> seg(k, g, len), writeind)
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
