using MAT


function proceed()
  path = "/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/low_rank_tensor_approx/low-rank-tensor-approximation/src/"
  include(path*"cumulants.jl")
  include(path*"read_pictures.jl")
  include(path*"moment_calc.jl")
  rmprocs(workers())
  addprocs()
  data, n = read_hyperspectral("indian_pines_hyperspectral.npy", 5)
  data = data/maximum(data)
  C2 = get_cumulant2(data)
  r, X2, S = cov_calc(C2, n)
  C3 = get_cumulant3(convert(Array{Float32,2},data))
  X3 = hosvd(C3).matrices[1]
  C4 = get_cumulant4(convert(Array{Float32,2},data))
  X4 = hosvd(C4).matrices[1]
  matwrite("cumulants.mat", Dict{Any,Any}(
      "C2" => C2,
      "C3" => C3,
      "C4" => C4,
      "X2" => X2,
      "X3" => X3,
      "X4" => X4
  ))
end
