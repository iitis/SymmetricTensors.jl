# ---- generated SymmetricTensors ----

"""
    randsymarray(::Type{T}, dim ::Int, N::Int = 4)

Return an ``N``-dimmensional random super-symmetric array with elements of type `T` drawn from a uniform distribution on ``[0, 1)``,
where `dim` denotes the data size.
"""
function randsymarray(::Type{T}, dim::Int, N::Int = 4) where T<:Real
  t = zeros(fill(dim, N)...,)
  for i in pyramidindices(N,dim)
    n = rand(T)
    for j in collect(permutations(i))
        @inbounds t[j...] = n
    end
  end
  t
end

"""
    randsymarray(dim::Int, N::Int = 4)

Return an ``N``-dimmensional random super-symmetric array with elements of type `Float64` drawn from a uniform distribution on ``[0, 1)``,
where `dim` denotes the data size.
"""
randsymarray(dim::Int, N::Int = 4) = randsymarray(Float64, dim, N)


"""
    fixpointperms(j::NTuple{N, Int})

Return a vector of vectors, which is a fixed-point permutation of given multi-indices.

!!! note
    This is a naive implementation.
"""
function fixpointperms(j::NTuple{N, Int}) where N
  r = []
  for p in permutations(1:N)
    if j == j[p]
      push!(r, p)
    end
  end
  r
end

"""
    randblock(::Type{T}, dims::NTuple{N, Int}, j::NTuple{N, Int})

Return a block of size `dims` and position `j` filled with a series of uniformly distributed random numbers
of type `T`.
"""
function randblock(::Type{T}, dims::NTuple{N, Int}, j::NTuple{N, Int}) where {T<:Real, N}
  t = zeros(dims)
  ofset = vcat([0], cumsum(counts([j...])))
  fp = fixpointperms(j)
  for i in 1:(prod(dims))
    i = Tuple(CartesianIndices(dims)[i])
    if mapreduce(k -> issorted(i[ofset[k]+1:ofset[k+1]]), *, 1:length(ofset)-1)
      n = rand(T)
      for k in fp
          @inbounds t[i[k]...] = n
      end
    end
  end
  t
end


"""
    rand(::Type{SymmetricTensor{T, N}}, n::Int, b::Int = 2)

Return an ``N``-dimmensional random `SymmetricTensor` with elements of type `T` drawn from a uniform distribution on ``[0, 1)``,
where `n` denotes the data size and `b` denotes the block size.
"""
function rand(::Type{SymmetricTensor{T, N}}, n::Int, b::Int = 2) where {T<:AbstractFloat, N}
  sizetest(n, b)
  nbar = mod(n,b)==0 ? n÷b : n÷b + 1
  ret = arraynarrays(Float64, fill(nbar, N)...,)
  for j in pyramidindices(N, nbar)
    dims = (mod(n,b) == 0 || !(nbar in j)) ? (fill(b,N)...,) : map(i -> (i == nbar) ? n - b*(nbar-1) : b, j)
    if j == (unique(j)...,)
      @inbounds ret[j...] = rand(T, dims...)
    else
      @inbounds ret[j...] = randblock(T, dims, j)
    end
  end
  SymmetricTensor(ret; testdatstruct = false)
end
