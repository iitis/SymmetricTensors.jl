module Boxtensors
  using NullableArrays
  using Iterators
  using Tensors
  import Base: trace, vec, vecnorm, +, -, *, .*, /, \, ./, size, transpose, convert


  # Type implementation ond simple operations
  include("boxstructure.jl")

  #calculates moments and cumulants
  include("cumulantsbs.jl")

  #advamced operations on bs, including some "Kolda implementations"
  include("operationsbs.jl")

  #generates random multivariate data sung copulas
  include("copulagen.jl")

  #naive algorithms for computation time tests
  include("seminaivecum.jl")


  export BoxStructure, convert, +, -, *, /, add, trace, vec, vecnorm, covbs, modemult, square, bcss,
  bcssclass, momentbs, centre, cumulants, clcopulagen, pbc, naivecumulant
end
