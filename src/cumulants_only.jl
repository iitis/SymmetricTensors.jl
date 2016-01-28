path = "/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/low_rank_tensor_approx/low-rank-tensor-approximation/src/"
include(path*"read_pictures.jl")
using NPZ

function cumulants(infile, outfile1, outfile2, outfile3)
  if nprocs() == 1
      addprocs()
  end
  
  @everywhere begin
  path = "/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/low_rank_tensor_approx/low-rank-tensor-approximation/src/"
  include(path*"Cumulants.jl")
      using Cumulants
  end

  data::Matrix{Float32} = read_hyperspectral(infile)
  data = data/maximum(data)
  C2 = cumulant2(data)
  C3 = cumulant3(data)
  C4 = cumulant4(data)
  
  npzwrite(outfile1, C2)
  npzwrite(outfile2, C3)
  npzwrite(outfile3, C4)

  end
  

cumulants("test.npy", "C2.npz", "C3.npz", "C4.npz")


