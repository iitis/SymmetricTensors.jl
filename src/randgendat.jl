# ---- generated SymmetricTensors ----

"""
  randsymarray(T, dim ::Int, N::Int = 4)

Returns N-dimmensional random super-symmetric array with elements of type T drawn from uniform distribution on [0,1),
dim denotes data size.

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
  randsymarray(T, dim::Int, N::Int = 4)

Returns N-dimmensional random super-symmetric array with Float64 elements drawn from uniform distribution on [0,1),
dim denotes data size.

"""
randsymarray(dim::Int, N::Int = 4) = randsymarray(Float64, dim, N)


"""

  fixpointperms(j::NTuple{N, Int}) where N

Returns Vactor{Vector}, a fix point permutation of given multiindex

TODO: this is a naive implementation
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

Returns a block of size dims and position j by a uniformly distributed random number
of type T

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
rand(SymmetricTensor{T, N}, n::Int, b::Int = 2)

Returns N-dimensional random SymmetricTensor with elements of type T drawn from uniform distribution on [0,1),
n denotes data size and b denotes block size.

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
