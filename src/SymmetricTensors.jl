VERSION >= v"0.5.0-dev+6521"

module SymmetricTensors
  using NullableArrays
  using Combinatorics
  import Base: +, -, *, /, size, convert, getindex, diag, indices, broadcast, rand

  # Type implementation and simple operations
  include("symmetrictensor.jl")
  #generates random SymmetricTensors
  include("randgendat.jl")

  export SymmetricTensor, convert, +, -, *, /, unfold, diag, rand
end
