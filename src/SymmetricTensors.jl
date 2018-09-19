module SymmetricTensors
  using Combinatorics
  using Base.Cartesian
  using StatsBase
  using Random
  using LinearAlgebra
  import Base: +, -, *, /, size, getindex, rand, setindex!

  const ArrayNArrays{T,N} = Array{Union{Array{T, N}, Nothing}, N} where {T<:AbstractFloat, N}
  function arraynarrays(T::Type, dims...)
      N = length(dims)
      symten = ArrayNArrays{T,N}(undef, dims...)
      fill!(symten, nothing)
      return symten
  end

  # Type implementation and simple operations
  include("symmetrictensor.jl")
  #generates random SymmetricTensors
  include("randgendat.jl")

  export SymmetricTensor, convert, +, -, *, /, unfold, diag, rand,
         ArrayNArrays, arraynarrays, setindex!
end
