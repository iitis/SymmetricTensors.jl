module Boxtensors
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
  include("boxstructure.jl")

  #calculates moments and cumulants
  include("cumulantsbs.jl")

  #generates random multivariate data sung copulas
  include("copulagen.jl")

  #naive algorithms for computation time tests
  include("naivecum.jl")


  export BoxStructure, convert, +, -, *, /, momentbs, centre, cumulants, clcopulagen, naivecumulant
end
