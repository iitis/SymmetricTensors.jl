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
      t[j...] = randn
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

"""
  rand{(::::Type{SymmetricTensor{T, N}}, dats::Int, N::Int = 4)

Returns N-dims random SymmetricTensor of Float64, size dats and blocksize - bls

"""

rand{N}(::Type{SymmetricTensor{N}}, dats::Int, bls::Int = 2) =
  rand(SymmetricTensor{Float64, N}, dats, bls)
