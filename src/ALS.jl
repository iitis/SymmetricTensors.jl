path = "/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/low_rank_tensor_approx/low-rank-tensor-approximation/src/"
include(path*"Cumulants.jl")
using Cumulants
using MAT
using Tensors
using NPZ

  
function calc(infile1, infile2, infile3, outfile)  
  
  C2::Matrix{Float32} = npzread(infile1)
  C3::Array{Float32} = npzread(infile2)
  C4::Array{Float32} = npzread(infile3)

  
  
  function phi_calc{T<:AbstractFloat}(C2::Matrix{T}, C3::Array{T}, C4::Array{T})
  k::Int32 = npzread("parameter.npy")
      w_2 = 1/2
      w_3 = 1/6
      w_4 = 1/24
      Uc = Tensors.left_singular_vectors(hcat(w_2*C2,w_3*Tensors.unfold(C3, 1),w_4*Tensors.unfold(C4, 1)))[:,1:k]
      U_f = []
      for i = 0:10
	  T3 = C3
	  for i = 2:3
	      T3 = Tensors.modemult(T3, Uc', i)
	  end
	  T4 = C4
	  for i = 2:4
	      T4 = Tensors.modemult(T4, Uc', i)
	  end
	  Uc = Tensors.left_singular_vectors(hcat(w_2*C2*Uc,w_3*Tensors.unfold(T3, 1),w_4*Tensors.unfold(T4, 1)))[:,1:k]
	  C_4 = C4
	  for i = 1:4
	      C_4 = Tensors.modemult(C_4, Uc', i)
	  end
	  C_3 = C3
	  for i = 1:3
	      C_3 = Tensors.modemult(C_3, Uc', i)
	  end
	  C_2 = Uc'*C2*Uc

	  println(norm_tensor(C_2)/norm_tensor(C2))
	  println(norm_tensor(C_3)/norm_tensor(C3))
	  println(norm_tensor(C_4)/norm_tensor(C4))
	  println("....next itter....")
	  U_f = Uc
      end
      U_f
  end

  function write(filename)
    U_f::Matrix{Float32} = phi_calc(C2, C3, C4);
    npzwrite(filename, U_f)
  end

  write(outfile)
  end

calc("C2.npz", "C3.npz", "C4.npz", "U_common.npy")  

