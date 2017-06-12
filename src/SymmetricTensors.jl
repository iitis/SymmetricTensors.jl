module SymmetricTensors
  using NullableArrays
  import Base: +, -, *, .*, /, ./, size, convert, getindex

  # Type implementation and simple operations
  include("symmetrictensor.jl")

  export SymmetricTensor, convert, +, -, *, .*, /, ./, unfold
end
