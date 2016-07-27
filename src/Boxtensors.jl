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

  #algorithms for tests
  #.....................
  include("seminaivecum.jl") #semi naive algorithm (using tensor networks, not using box structure)
  include("fdcumulants.jl") #forwarddiff using definition


  export BoxStructure, convert, +, -, *, /, add, trace, vec, vecnorm, covbs, modemult, square, bcss,
  bcssclass, momentbs, centre, cumulants, clcopulagen, snaivecumulant, get_diff, moment3, moment4, pbc
end
