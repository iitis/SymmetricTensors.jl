module SymmetricTensors
  using NullableArrays
  import Base: +, -, *, .*, /, \, ./, size, convert, getindex

  # Type implementation and simple operations
  include("symmetrictensors.jl")

  export SymmetricTensor, convert, +, -, *, .*, /, \, ./, unfold
end
