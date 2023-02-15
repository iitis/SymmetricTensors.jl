"""
    SymmetricTensor(frame, bls, bln, dats, sqr)

Construct a symmetric tensor.

# Arguments
- `frame::ArrayNArrays{T,N}`: the raw data.
- `bls::Int`: the size of ordinary block (the same in each direction).
- `bln::Int`: the number of blocks.
- `dats::Int`: the size of data stored (the same in each direction).
- `sqr::Bool`: if last blocks are squares (N-squares).
"""
mutable struct SymmetricTensor{T <: AbstractFloat, N}
    frame::ArrayNArrays{T,N}
    bls::Int
    bln::Int
    dats::Int
    sqr::Bool
    function (::Type{SymmetricTensor})(frame::ArrayNArrays{T,N};
        testdatstruct::Bool = true) where {T, N}
        bls = size(frame[fill(1, N)...,], 1)
        bln = size(frame, 1)
        last_block = size(frame[end], 1)
        dats = bls * (bln - 1) + last_block
        if testdatstruct
          frtest(frame)
        end
        new{T, N}(frame, bls, bln, dats, bls == last_block)
    end
end

"""
    unfold(ar::Array{T,N}, mode::Int)

Return a matrix being an unfold of `N` dimensional array in a given mode.

```jldoctest
julia> A = reshape(collect(1.:8.), 2, 2, 2);

julia> unfold(A, 1)
2×4 Array{Float64,2}:
 1.0  3.0  5.0  7.0
 2.0  4.0  6.0  8.0

julia> unfold(A, 2)
2×4 Array{Float64,2}:
 1.0  2.0  5.0  6.0
 3.0  4.0  7.0  8.0

julia> unfold(A, 3)
2×4 Array{Float64,2}:
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
```
"""
function unfold(ar::Array{T,N}, mode::Int) where {T <: Real, N}
    C = [1:mode-1; mode+1:N]
    i = size(ar)
    k = prod(i[C])
    return reshape(permutedims(ar,[mode; C]), i[mode], k)
end


"""
    issymetric(ar::Array, atol::Float64)

Throw an `AssertionError` if `ar` is not symmetric under a given tolerance.

```jldoctest
julia> A = reshape(collect(1.:8.), 2, 2, 2);

julia> julia> issymetric(A)
ERROR: AssertionError: not symmetric
```
"""
function issymetric(ar::Array{T, N}, atol::Float64 = 1e-7) where {T <: AbstractFloat, N}
  for i=2:N
     maximum(abs.(unfold(ar, 1)-unfold(ar, i))) < atol ||throw(AssertionError("not symmetric"))
  end
end

"""
    frtest(data::ArrayNArrays{T,N})

Throw an `AssertionError` if:
- all sizes of nullable array are not equal, or
- some undergiagonal blocks are not null, or
- some blocks (not the last ones) are not squares (N-squares), or
- some diagonal blocks are not symmetric.
"""
function frtest(data::ArrayNArrays{T,N}) where {T <: AbstractFloat, N}
  bln = size(data, 1)
  bls = size(data[fill(1, N)...,], 1)
  all(collect(size(data)) .== bln) || throw(AssertionError("frame not has non-equal dimensions"))

  for i in CartesianIndices(size(data))
      if data[i]!=nothing && !issorted(i.I)
          throw(AssertionError("underdiagonal block not null"))
      end
  end

  for i in pyramidindices(N, bln-1)
    @inbounds all(collect(size(data[i...])) .== bls)||
        throw(AssertionError("$i block not square"))
  end
  for i=1:bln
    @inbounds issymetric(data[fill(i, N)...,])
  end
end

"""
    pyramidindices(dims::Int, tensize::Int)

```jldoctest
julia> pyramidindices(2,3)
6-element Array{Tuple{Int64,Int64},1}:
 (1,1)
 (1,2)
 (1,3)
 (2,2)
 (2,3)
 (3,3)
```
"""
function pyramidindices(dims::Int, tensize::Int)
    multinds = Tuple{fill(Int,dims)...,}[]
    @eval begin
        @nloops $dims i x -> (x==$dims) ? (1:$tensize) : (i_{x+1}:$tensize) begin
            @inbounds multind = @ntuple $dims x -> i_{$dims-x+1}
            push!($multinds, multind)
        end
    end
    multinds
end

"""
    pyramidindices(st::SymmetricTensor)

Return the indices of the unique elements of the given symmetric tensor.
"""
pyramidindices(st::SymmetricTensor{<:Any, N}) where N = pyramidindices(N, st.dats)

"""
    sizetest(dats::Int, bls::Int)

Throw a `DimensionMismatch` if the block size is grater than the data size.

```jldoctest
julia> SymmetricTensors.sizetest(2,3)
ERROR: DimensionMismatch("bad block size 3 > 2")
```
"""
sizetest(dats::Int, bls::Int) =
  (dats >= bls > 0)|| throw(DimensionMismatch("bad block size $bls > $dats"))

"""
    getblockunsafe(st::SymmetricTensor, i::Tuple)

Return a block from `st`.
    
It is unsafe and works only if multi-indices are sorted.
"""
getblockunsafe(st::SymmetricTensor, mulind::Tuple) = st.frame[mulind...]


"""
    getblock(st::SymmetricTensor, i::Tuple)

Return a block from `st`.
        
It also works for multi-indices which are not sorted.
"""
function getblock(st::SymmetricTensor, mulind::Tuple)
  ind = sortperm([mulind...])
  permutedims(getblockunsafe(st, mulind[ind]), invperm(ind))
end

"""
    size(st::SymmetricTensor)

Return the size of a symmetric tensor.
"""
size(st::SymmetricTensor{<:Any, N}) where N = ntuple(i -> st.dats, N)

"""
    getindex(st::SymmetricTensor, mulind::Tuple)

Return the element in `st` with a given multi-index.
"""
function getindex(st::SymmetricTensor, mulind::Int...)
  b = st.bls
  j = map(k -> div((k-1), b)+1, mulind)
  i = map(k -> ((k-1)%b)+1, mulind)
  getblock(st, j)[i...]
end

"""
    setindexunsafe!(st::SymmetricTensor{T,N}, x::T, mulind::Int...)

Change the value at the given multi-index of `st` unsafely.
"""
function setindexunsafe!(st::SymmetricTensor{T,N}, x::T,  mulind::Int...) where {T <: AbstractFloat, N}
    b = st.bls
    j = map(k -> div((k-1), b)+1, mulind)
    i = map(k -> ((k-1)%b)+1, mulind)
    if j == (unique(j)...,)
        getblockunsafe(st, j)[i...] = x
    else
        for p in fixpointperms(j)
            getblockunsafe(st, j)[i[p]...] = x
        end
    end
end

"""
    setindex!(st::SymmetricTensor{T,N}, x::T,  mulind::Int...)

Change the value at the given multi-index of `st` if possible.
"""
function setindex!(st::SymmetricTensor{T,N}, x::T,  mulind::Int...) where {T <: AbstractFloat, N}
    setindexunsafe!(st, x, sort([mulind...])...)
end

"""
    ind2range(i::Int, bls::Int, dats::Int)

Return a range given index `i`, size of a block `bls`, and size of data `dats`.

```jldoctest
julia> ind2range(2, 3, 5)
4:5
```
"""
ind2range(i::Int, bls::Int, dats::Int) = (i-1)*bls+1: ((i*bls <= dats) ? i*bls : dats)

"""
    SymmetricTensor(data::Array, bls::Int)

Construct a `SymmetricTensor` from an `Array`.

```jldoctest
julia> a = reshape(collect(1.:16.), 4, 4);

julia> SymmetricTensor(a*a')
SymmetricTensors.SymmetricTensor{Float64,2}(Union{Array{Float64,2}, Void}[[276.0 304.0; 304.0 336.0][332.0 360.0; 368.0 400.0]; nothing [404.0 440.0; 440.0 480.0]], 2, 2, 4, true)
```
"""
function SymmetricTensor(data::Array{T, N}, bls::Int = 2) where {T <: AbstractFloat, N}
  issymetric(data)
  dats = size(data,1)
  sizetest(dats, bls)
  bln = mod(dats,bls)==0 ?  dats÷bls : dats÷bls + 1
  symten = arraynarrays(T, fill(bln, N)...,)
  for writeind in pyramidindices(N, bln)
      readind = map(k::Int -> ind2range(k, bls, dats), writeind)
      @inbounds symten[writeind...] = data[readind...]
  end
  SymmetricTensor(symten)
end

"""
    Array(st::SymmetricTensor{T,N})

Return an `N` dimensional array converted from a `SymmetricTensor`.
"""
function Array(st::SymmetricTensor{T,N}) where {T<:AbstractFloat, N}
  array = zeros(T, fill(st.dats, N)...,)
  for i = 1:(st.bln^N)
    dims = (fill(st.bln, N)...,)
    readind= Tuple(CartesianIndices(dims)[i])
    writeind = map(k -> ind2range(readind[k], st.bls, st.dats), 1:N)
    @inbounds array[writeind...] = getblock(st, readind)
  end
  array
end
# ---- basic operations on Symmetric Tensors ----

"""
    diag(st::SymmetricTensor)

Return the superdiagonal of `st`.
"""
diag(st::SymmetricTensor{T,N}) where {T<: AbstractFloat, N} = map(i->st[fill(i, N)...,], 1:st.dats)


# implements simple operations on bs structure


for f = (:+, :-)
  @eval function ($f)(st::SymmetricTensor{T,N}...) where {T <: AbstractFloat, N}
    for s in st[2:end]
      s.dats == st[1].dats || throw(DimensionMismatch("dimensions must match"))
    end
    narg = size(st, 1)
    stret = similar(st[1].frame)
    for i in pyramidindices(N, st[1].bln)
      @inbounds stret[i...] = broadcast($f, map(t -> getblockunsafe(t, i), st)...)
    end
    SymmetricTensor(stret; testdatstruct = false)
  end
end

for f = (:+, :-, :*, :/)
  @eval function ($f)(st::SymmetricTensor{T, N}, numb::S) where {T <: AbstractFloat, S <: Real, N}
      stret = similar(st.frame)
      for i in pyramidindices(N, st.bln)
        @inbounds stret[i...] = broadcast($f, getblockunsafe(st, i), numb)
      end
      SymmetricTensor(stret; testdatstruct = false)
  end
end


for f = (:+, :*)
  @eval ($f)(numb::S, st::SymmetricTensor{T}) where {T <: AbstractFloat, S <: Real} =
  ($f)(st::SymmetricTensor{T}, numb::S)
end
