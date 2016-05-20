module Test
  using Base.Test
  using NullableArrays
  using SymmetricMatrix
  using Iterators
  using Tensors
  importall SymmetricMatrix
  
  symmetrise{T <: AbstractFloat}(matrix::Matrix{T}) = matrix*transpose(matrix)
  
  function generatedata(seed::Int = 1234, n::Int = 20, seg::Int = 6, l::Int = 1000)
      srand(seed)
      rmat = randn(n,n)
      rmat, symmetrise(rmat), convert(BoxStructure{Float64}, symmetrise(rmat), seg), randn(l,n), bitrand(n,n), (im*rmat + rmat)*(im*rmat + rmat)'
   end
   
   function createsegments(randarray, nonullel::Bool = false, s1::Int = 4)
   dims = ndims(randarray)
      structure = NullableArray(Array{Float64, dims}, fill(s1, dims)...)
      for i in product(fill(1:s1, dims)...)
	 issorted(i)? structure[i...] = randarray : ()
      end
      if nonullel
	structure[reverse(collect(1:dims))...] = randarray
      end
      structure
  end
    
    m, sm, smseg, data, boolean, comlx = generatedata()
    smseg1 = convert(BoxStructure{Float64}, sm, 2)
    m2, sm2, smseg2 = generatedata(1233)
    badsegments = 7

    @test_approx_eq(convert(Array{Float64}, (smseg)), sm)
    @test_approx_eq(convert(Array{Float16},convert(BoxStructure{Float16}, Matrix{Float16}(sm), 5)), Matrix{Float16}(sm))
    @test_approx_eq(convert(Array{Float32},convert(BoxStructure{Float32}, Matrix{Float32}(sm), 5)), Matrix{Float32}(sm))
    @test_approx_eq(convert(Array{AbstractFloat},convert(BoxStructure{AbstractFloat}, Matrix{AbstractFloat}(sm), 5)), Matrix{AbstractFloat}(sm))
    
    @test_approx_eq(smseg*smseg2, sm*sm2)
    @test_approx_eq(convert(Array{Float64},smseg+smseg2), sm+sm2)
    @test_approx_eq(convert(Array{Float64},smseg-smseg2), sm-sm2)
    @test_approx_eq(convert(Array{Float64},smseg.*smseg2), sm.*sm2)
    @test_approx_eq(convert(Array{Float64},smseg./smseg2), sm./sm2)
    @test_approx_eq(smseg*m, sm*m)
    
    @test_approx_eq(convert(Array{Float64},smseg*2.1), sm*2.1)
    @test_approx_eq(convert(Array{Float64},smseg/2.1), sm/2.1)
    @test_approx_eq(convert(Array{Float64},smseg*2), sm*2)
    @test_approx_eq(convert(Array{Float64},smseg+2.1), sm+2.1)
    @test_approx_eq(convert(Array{Float64},smseg-2.1), sm-2.1)
    @test_approx_eq(convert(Array{Float64},smseg+2), sm+2)

    @test_approx_eq(smseg*m[:,1:12], sm*m[:,1:12])
    @test_approx_eq(smseg*m[:,1:7], sm*m[:,1:7])
    @test_approx_eq(smseg*m[:,1:1], sm*m[:,1:1])
    @test_approx_eq(vec(smseg), vec(sm))
    @test_approx_eq(trace(smseg), trace(sm))
    @test_approx_eq(vecnorm(smseg), vecnorm(sm))
    @test_approx_eq(convert(Array{Float64},square(smseg)), sm*sm)
    @test_approx_eq(convert(Array{Float64},covbs(data, smseg.sizesegment, false)), cov(data, corrected=false))
    @test_approx_eq(convert(Array{Float32}, covbs(Matrix{Float32}(data), smseg.sizesegment, false)), cov(Matrix{Float32}(data), corrected=false))
    @test_approx_eq(convert(Array{Float64},covbs(data, smseg.sizesegment, true)), cov(data, corrected=true))
    
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, m, 5))
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, sm[:,1:15], 5))
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, sm,  badsegments))
    @test_throws(MethodError, convert(BoxStructure{Float64}, boolean, 5))
    @test_throws(TypeError, convert(BoxStructure{Bool}, boolean, 5))
    @test_throws(MethodError, convert(BoxStructure{Float64}, comlx, 5))
    @test_throws(TypeError, convert(BoxStructure{Complex64}, comlx, 5))
    
    
    @test_throws(DimensionMismatch, BoxStructure(smseg.frame[:,1:2]))
    @test_throws(DimensionMismatch, BoxStructure(createsegments(sm[:,1:2])))
    @test_throws(DimensionMismatch, BoxStructure(createsegments(m)))
    @test_throws(ArgumentError, BoxStructure(createsegments(sm, true)))
    @test_throws(DimensionMismatch, smseg*smseg1)
    @test_throws(DimensionMismatch, smseg+smseg1)
    @test_throws(DimensionMismatch, smseg.*smseg1)
    @test_throws(DimensionMismatch, smseg*(m[1:5,:]))
    @test_throws(DimensionMismatch, covbs(data,  badsegments))
    
    function genstensor(T::Type, dims::Int, l::Int, seed::Int = 1234)
      srand(seed)
      ret = zeros(T, fill(l, dims)...)
      indices = collect(product(fill(collect(1:l), dims)...))
      elements = Array{T,1}(randn(l^dims÷dims))
      r = 1
      for i in indices 
	  if issorted(i)          
	      for k in collect(permutations(i))
		  ret[k...] = elements[r]
	      end
	      r += 1
	  end
      end
      ret
    end
    
    function randtensor(T::Type, dims::Int, l::Int, seed::Int = 1234)
	srand(seed)
	if T <: AbstractFloat
	  return Array{T,dims}(randn(fill(l, dims)...))
	elseif T <: Bool
	  return bitrand(fill(l, dims)...)
	elseif T <: Complex
	  return Array{T,dims}(randn(fill(l, dims)...) + im*rand(fill(l, dims)...))
	end
	  
    end
    
    stensor = genstensor(Float64, 4,10)
    bstensor = (convert(BoxStructure{Float64}, stensor, 3))
    stensor1 = genstensor(Float64, 4,10, 1233)
    bstensor1 = (convert(BoxStructure{Float64}, stensor1, 3))
    stensor2 = genstensor(Float64, 4,11)
    bstensor2 = (convert(BoxStructure{Float64}, stensor2, 3))
    stensor3 = genstensor(Float64, 3,10)
    bstensor3 = (convert(BoxStructure{Float64}, stensor2, 3))
    
    # tests for tensors
    
    @test_approx_eq(convert(Array{Float64}, (bstensor)), stensor)
    @test_approx_eq(convert(Array{Float16},convert(BoxStructure{Float16}, Array{Float16}(stensor), 3)), Array{Float16}(stensor))
    @test_approx_eq(convert(Array{Float32},convert(BoxStructure{Float32}, Array{Float32}(stensor), 3)), Array{Float32}(stensor))
    @test_approx_eq(convert(Array{AbstractFloat},convert(BoxStructure{AbstractFloat}, Array{AbstractFloat}(stensor), 3)), Array{AbstractFloat}(stensor))
    
    @test_approx_eq(convert(Array{Float64},bstensor*2.1), stensor*2.1)
    @test_approx_eq(convert(Array{Float64},bstensor/2.1), stensor/2.1)
    @test_approx_eq(convert(Array{Float64},bstensor*2), stensor*2)
    @test_approx_eq(convert(Array{Float64},bstensor+2.1), stensor+2.1)
    @test_approx_eq(convert(Array{Float64},bstensor-2.1), stensor-2.1)
    @test_approx_eq(convert(Array{Float64},bstensor+2), stensor+2)
    @test_approx_eq(vec(bstensor), vec(stensor))
    @test_approx_eq(convert(Array{Float64},bstensor+bstensor1), stensor+stensor1)
    @test_approx_eq(convert(Array{Float64},bstensor-bstensor1), stensor-stensor1)
    @test_approx_eq(convert(Array{Float64},bstensor.*bstensor1), stensor.*stensor1)
    @test_approx_eq(convert(Array{Float64},bstensor./bstensor1), stensor./stensor1)
    @test_approx_eq(modemult(bstensor, m[:,1:size(stensor, 1)], 1), Tensors.modemult(stensor, m[:,1:size(stensor, 1)], 1))
    @test_approx_eq(modemult(bstensor, m[:,1:size(stensor, 1)], 2), Tensors.modemult(stensor, m[:,1:size(stensor, 1)], 2))
    @test_approx_eq(modemult(bstensor, m[:,1:size(stensor, 1)], 4), Tensors.modemult(stensor, m[:,1:size(stensor, 1)], 4))
    
    @test_throws(DimensionMismatch, modemult(bstensor, m[:,1:2], 4))
    @test_throws(DimensionMismatch, modemult(bstensor, m[:,1:size(stensor, 1)], ndims(stensor)+1))
    
    @test_throws(DimensionMismatch, bstensor+bstensor2)
    @test_throws(DimensionMismatch, bstensor.*bstensor2)
    @test_throws(DimensionMismatch, bstensor+bstensor3)
    @test_throws(DimensionMismatch, bstensor.*bstensor3)

    @test_throws(DimensionMismatch, BoxStructure(bstensor.frame[:,:,:,1:2]))
    @test_throws(DimensionMismatch, BoxStructure(createsegments(randtensor(Float64, 4, 10))))
    @test_throws(DimensionMismatch, BoxStructure(createsegments(stensor[:,:,:,1:2])))
    @test_throws(ArgumentError, BoxStructure(createsegments(stensor, true)))
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, stensor,  badsegments))
    
    @test_throws(MethodError, convert(BoxStructure{Float64}, randtensor(Bool, 4, 10), 3))
    @test_throws(TypeError, convert(BoxStructure{Bool}, randtensor(Bool, 4, 10), 3))
    
    @test_throws(MethodError, convert(BoxStructure{Float64}, randtensor(Complex64, 4, 10), 3))
    @test_throws(TypeError, convert(BoxStructure{Complex64}, randtensor(Complex64, 4, 10), 3))

    

    
    
  
end
