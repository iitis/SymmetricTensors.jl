VERSION >= v"0.5.0-dev+6521"

module SymmetricTensors
  using NullableArrays
  import Base: +, -, *, .*, /, ./, size, convert, getindex, diag

  # Type implementation and simple operations
  include("symmetrictensor.jl")

  export SymmetricTensor, convert, +, -, *, .*, /, ./, unfold, diag
end
