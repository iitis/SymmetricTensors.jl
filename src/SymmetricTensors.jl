module SymmetricTensors
  using NullableArrays
  using Iterators
  using Tensors #Gawron tensors
  using ForwardDiff
  if VERSION >= v"0.5.0-dev+1204"
    using Combinatorics
  end
  import Base: trace, vec, vecnorm, +, -, *, .*, /, \, ./, size, transpose, convert, ndims
  import Tensors: unfold


  # Type implementation ond simple operations
  include("symmetrictensors.jl")

  #calculates moments and cumulants
  include("cumulants.jl")

  export SymmetricTensor, convert, +, -, *, /, momentbs, center, cumulants
end
