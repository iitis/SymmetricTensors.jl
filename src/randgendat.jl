# ---- generated SymmetricTensors ----

"""
  randsymarray(::Type{T}, dats::Int, N::Int = 4)

Returns N-dims super-symmetric array of T type numbers and sizes dats

"""

function randsymarray(::Type{T}, dats::Int, N::Int = 4) where T<:Real
  t = zeros(fill(dats, N)...)
  for i in indices(N,dats)
    randn = rand(T)
    for j in collect(permutations(i))
      @inbounds t[j...] = randn
    end
  end
  t
end

"""
  randsymarray(::Type{T}, dats::Int, N::Int = 4)

Returns N-dims super-symmetric array of Float64 and sizes dats

"""

randsymarray(dats::Int, N::Int = 4) = randsymarray(Float64, dats, N)

"""
  rand(SymmetricTensor{T, N}, dim::Int, bls::Int = 2)

Returns N-dimmensional random SymmetricTensor with elements of type T drawn from uniform distribution on [0,1), 
dim denotes data size and bls denotes block size.

"""

rand(::Type{SymmetricTensor{T, N}}, dim::Int, bls::Int = 2) where {T<:AbstractFloat, N} =
  convert(SymmetricTensor, randsymarray(T, dim, N), bls)
