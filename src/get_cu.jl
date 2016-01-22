include(joinpath(path,"cumulants.jl"))
path = "/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/low_rank_tensor_approx/low-rank-tensor-approximation/src/"
include(joinpath(path,"read_pictures.jl"))
using Cumulants
using MAT

function calc_cumulants(infile, outfile)
  if nprocs()==1
    addprocs()
  end
  data::Matrix{Float32} = read_hyperspectral(infile)
  data = data/maximum(data)
  C2 = cumulant2(data)
  X2 = hosvd(C2).matrices[1]
  C3 = cumulant3(data)
  X3 = hosvd(C3).matrices[1]
  C4 = cumulant4(data)
  X4 = hosvd(C4).matrices[1]
  matwrite(outfile, Dict{Any,Any}(
      "C2" => C2,
      "C3" => C3,
      "C4" => C4,
      "X2" => X2,
      "X3" => X3,
      "X4" => X4
  ))
end

calc_cumulants("test.npy", "cumulants.mat")
