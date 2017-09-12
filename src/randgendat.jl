# ---- generated SymmetricTensors ----

"""
  randsymarray{T<:Real}(::Type{T}, dats::Int, N::Int = 4)

Returns N-dims super-symmetric array of T type numbers and sizes dats

"""

function randsymarray{T<:Real}(::Type{T}, dats::Int, N::Int = 4)
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
  randsymarray{T<:Real}(::Type{T}, dats::Int, N::Int = 4)

Returns N-dims super-symmetric array of Float64 and sizes dats

"""

randsymarray(dats::Int, N::Int = 4) = randsymarray(Float64, dats, N)

"""
  rand{(::::Type{SymmetricTensor{T, N}}, dats::Int, N::Int = 4)

Returns N-dims random SymmetricTensor of T type numbers, size dats and blocksize - bls

"""

rand{T<:AbstractFloat, N}(::Type{SymmetricTensor{T, N}}, dats::Int, bls::Int = 2) =
  convert(SymmetricTensor, randsymarray(T, dats, N), bls)
