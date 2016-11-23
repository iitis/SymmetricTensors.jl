module SymmetricTensors
  using NullableArrays
  import Base: +, -, *, .*, /, \, ./, size, convert

  # Type implementation ond simple operations
  include("symmetrictensors.jl")

  export SymmetricTensor, convert, +, -, *, .*, /, \, ./, unfold
end
