module SymmetricTensors
  using NullableArrays
  using Iterators
  using ForwardDiff
  if VERSION >= v"0.5.0-dev+1204"
    using Combinatorics
  end
  import Base: trace, vec, vecnorm, +, -, *, .*, /, \, ./, size, transpose, convert, ndims

  if VERSION >= v"0.5.0-dev+1204"
    include("constructorj5.jl")
  else
    include("constructorj4.jl")
  end

  # Type implementation ond simple operations
  include("symmetrictensors.jl")

  #calculates moments and cumulants
  include("cumulants.jl")

  #partitions. Knuth modified algorithm
  #include("/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/calc&codes/indpart/part.jl")

  export SymmetricTensor, convert, +, -, *, /, momentbs, center, cumulants, unfold ,outerp
end
