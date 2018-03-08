# ---- generated SymmetricTensors ----

"""
  randsymarray(T, dim ::Int, N::Int = 4)

Returns N-dimmensional random super-symmetric array with elements of type T drawn from uniform distribution on [0,1),
dim denotes data size.

"""

function randsymarray(::Type{T}, dim::Int, N::Int = 4) where T<:Real
  t = zeros(fill(dim, N)...)
  for i in _indices(N,dim)
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
