# ---- generated SymmetricTensors ----

"""
  randsymarray(T, dim ::Int, N::Int = 4)

Returns N-dimmensional random super-symmetric array with elements of type T drawn from uniform distribution on [0,1),
dim denotes data size.

"""

function randsymarray(::Type{T}, dim::Int, N::Int = 4) where T<:Real
  t = zeros(fill(dim, N)...)
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
  rand(SymmetricTensor{T, N}, dim::Int, bls::Int = 2)

Returns N-dimmensional random SymmetricTensor with elements of type T drawn from uniform distribution on [0,1),
dim denotes data size and bls denotes block size.

"""

rand(::Type{SymmetricTensor{T, N}}, dim::Int, bls::Int = 2) where {T<:AbstractFloat, N} =
  convert(SymmetricTensor, randsymarray(T, dim, N), bls)


"""
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

TODO: this is a naive implementation
"""
function randblock(::Type{T}, dims::NTuple{N, Int}, j::NTuple{N, Int}) where {T<:Real, N}
  t = zeros(dims)
  for i in 1:(prod(dims))
    i = ind2sub(dims, i)
    n = rand(T)
    for k in fixpointperms(j)
        @inbounds t[i[k]...] = n
    end
  end
  t
end

"""

"""
function randnn(::Type{T}, m::Int, n::Int, b::Int=2) where T<:AbstractFloat
  sizetest(n, b)
  nbar = mod(n,b)==0 ? n÷b : n÷b + 1
  ret = arraynarrays(Float64, fill(nbar, m)...)
  for j in pyramidindices(m, nbar)
    dims = (mod(n,b) == 0 || !(nbar in j))? (fill(b,m)...): map(i -> (i == nbar)? n - b*(nbar-1): b, j)
    jbar = (unique(j)...)
    r = (j== jbar)? rand(T, dims...): randblock(T, dims, j)
    @inbounds ret[j...] = r
  end
  SymmetricTensor(ret; testdatstruct = true)
end
