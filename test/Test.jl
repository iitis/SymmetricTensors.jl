module Test
  using Base.Test
  using NullableArrays
  using SymmetricTensors
  using Iterators
  using Distributions
  using ForwardDiff
  importall SymmetricTensors
  import Base: gradient

  #generates random multivariate data sung copulas
  include("copulagen.jl")

  #naive algorithms for computation time tests
  include("naivecum.jl")

  #semi naive algorithm for higher cumulants
  include("s_naive.jl")

  #forward diff
  include("fdcumulants.jl")

  symmetrise{T <: AbstractFloat}(matrix::Matrix{T}) = matrix*transpose(matrix)

  rmat = symmetrise(randn(6,6))
  converttest = convert(SymmetricTensor, rmat)
  @test_approx_eq(converttest.frame[1,1].value, rmat[1:3, 1:3])
  @test_approx_eq(converttest.frame[1,2].value, rmat[1:3, 4:6])
  @test_approx_eq(converttest.frame[2,2].value, rmat[4:6, 4:6])
  @test(isnull(converttest.frame[2,1]))


  function generatedata(seed::Int = 1234, n::Int = 15, seg::Int = 4, l::Int = 1000)
      srand(seed)
      rmat = randn(n,n)
      rmat, symmetrise(rmat), convert(SymmetricTensor, symmetrise(rmat), seg), rand(l,n), bitrand(n,n), (im*rmat + rmat)*(im*rmat + rmat)'
   end

   function createsegments(randarray, nonullel::Bool = false,  s1::Int = 4, addnonosymmbox::Bool = false)
   dims = ndims(randarray)
      structure = NullableArray(Array{Float64, dims}, fill(s1, dims)...)
      for i in product(fill(1:s1, dims)...)
	       issorted(i)? structure[i...] = randarray : ()
      end
      if nonullel
	       structure[reverse(collect(1:dims))...] = randarray
      end
      if addnonosymmbox
	       structure[fill(1, dims-1)...,2] = randarray[1:2,:]
      end
      structure
  end

    m, sm, smseg, data, boolean, comlx = generatedata()
    smseg1 = convert(SymmetricTensor, sm, 2)
    m2, sm2, smseg2 = generatedata(1233)
    badsegments = 8

    @test_approx_eq(convert(Array, smseg), sm)
    @test_approx_eq(convert(Array,convert(SymmetricTensor, Matrix{Float16}(sm), 5)), Matrix{Float16}(sm))
    @test_approx_eq(convert(Array,convert(SymmetricTensor, Matrix{Float32}(sm), 5)), Matrix{Float32}(sm))
    @test_approx_eq(convert(Array,convert(SymmetricTensor, Matrix{AbstractFloat}(sm), 5)), Matrix{AbstractFloat}(sm))

    @test_approx_eq(convert(Array,smseg+smseg2), sm+sm2)
    @test_approx_eq(convert(Array,smseg-smseg2), sm-sm2)
    @test_approx_eq(convert(Array,smseg.*smseg2), sm.*sm2)
    @test_approx_eq(convert(Array,smseg./smseg2), sm./sm2)

    @test_approx_eq(convert(Array,smseg*2.1), sm*2.1)
    @test_approx_eq(convert(Array,smseg/2.1), sm/2.1)
    @test_approx_eq(convert(Array,smseg*2), sm*2)
    @test_approx_eq(convert(Array,smseg+2.1), sm+2.1)
    @test_approx_eq(convert(Array,smseg-2.1), sm-2.1)
    @test_approx_eq(convert(Array,smseg+2), sm+2)

    @test_throws(AssertionError, convert(SymmetricTensor, m, 5))
    @test_throws(DimensionMismatch, convert(SymmetricTensor, sm[:,1:13], 6))
    @test_throws(DimensionMismatch, convert(SymmetricTensor, sm,  badsegments))
    @test_throws(MethodError, convert(SymmetricTensor, boolean, 5))
    @test_throws(MethodError, convert(SymmetricTensor, comlx, 5))
    @test_throws(AssertionError, SymmetricTensor(smseg.frame[:,1:2]))
    @test_throws(AssertionError, SymmetricTensor(createsegments(sm, false, 4, true)))
    @test_throws(AssertionError, SymmetricTensor(createsegments(m)))
    @test_throws(AssertionError, SymmetricTensor(createsegments(sm, true)))
    @test_throws(DimensionMismatch, smseg+smseg1)
    @test_throws(DimensionMismatch, smseg.*smseg1)


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
    bstensor = (convert(SymmetricTensor, stensor, 3))
    stensor1 = genstensor(Float64, 4,10, 1233)
    bstensor1 = (convert(SymmetricTensor, stensor1, 3))
    stensor2 = genstensor(Float64, 4,11)
    bstensor2 = (convert(SymmetricTensor, stensor2, 3))
    stensor3 = genstensor(Float64, 3,10)
    bstensor3 = (convert(SymmetricTensor, stensor2, 3))

    # tests for tensors

    @test_approx_eq(convert(Array, (bstensor)), stensor)
    @test_approx_eq(convert(Array,convert(SymmetricTensor, Array{Float16}(stensor), 3)), Array{Float16}(stensor))
    @test_approx_eq(convert(Array,convert(SymmetricTensor, Array{Float32}(stensor), 3)), Array{Float32}(stensor))
    @test_approx_eq(convert(Array,convert(SymmetricTensor, Array{AbstractFloat}(stensor), 3)), Array{AbstractFloat}(stensor))

    @test_approx_eq(convert(Array,bstensor*2.1), stensor*2.1)
    @test_approx_eq(convert(Array,bstensor/2.1), stensor/2.1)
    @test_approx_eq(convert(Array,bstensor*2), stensor*2)
    @test_approx_eq(convert(Array,bstensor+2.1), stensor+2.1)
    @test_approx_eq(convert(Array,bstensor-2.1), stensor-2.1)
    @test_approx_eq(convert(Array,bstensor+2), stensor+2)
    @test_approx_eq(convert(Array,bstensor+bstensor1), stensor+stensor1)
    @test_approx_eq(convert(Array,bstensor-bstensor1), stensor-stensor1)
    @test_approx_eq(convert(Array,bstensor.*bstensor1), stensor.*stensor1)
    @test_approx_eq(convert(Array,bstensor./bstensor1), stensor./stensor1)
    @test_throws(DimensionMismatch, bstensor+bstensor2)
    @test_throws(DimensionMismatch, bstensor.*bstensor2)
    @test_throws(DimensionMismatch, bstensor+bstensor3)
    @test_throws(DimensionMismatch, bstensor.*bstensor3)
    @test_throws(AssertionError, SymmetricTensor(bstensor.frame[:,:,:,1:2]))
    @test_throws(AssertionError, SymmetricTensor(createsegments(randtensor(Float64, 4, 10))))
    @test_throws(AssertionError, SymmetricTensor(createsegments(sm, false, 4, true)))
    @test_throws(AssertionError, SymmetricTensor(createsegments(stensor, true)))
    @test_throws(DimensionMismatch, convert(SymmetricTensor, stensor,  badsegments))
    @test_throws(MethodError, convert(SymmetricTensor, randtensor(Bool, 4, 10), 3))
    @test_throws(MethodError, convert(SymmetricTensor, randtensor(Complex64, 4, 10), 3))
    @test_approx_eq_eps(sum(abs(mean(center(m[1:3, 1:10]), 1))), 0, 1e-15)

  #rests moments via semi naive algorithms
  dat = center(data[1:15,1:5])
  @test_approx_eq(convert(Array, momentbs(dat, 3, 2)), moment3(dat))
  @test_approx_eq(convert(Array, momentbs(dat, 4, 2)), moment4(dat))

  dat1 = clcopulagen(10, 4)
  dat2 = dat1[:,1:3]
  dat3 = dat1[:,1:2]

  # test the bs algorithm using the naive (for square last block)
  cn = [naivecumulant(dat1, i) for i = 2:6]
  c2, c3, c4, c5, c6 = cumulants(6, dat1, 2)
  @test_approx_eq(convert(Array, c2),cn[1])
  @test_approx_eq(convert(Array, c3),cn[2])
  @test_approx_eq(convert(Array, c4),cn[3])
  @test_approx_eq(convert(Array, c5),cn[4])
  @test_approx_eq(convert(Array, c6),cn[5])

 # for nonsquare last block
  c2, c3, c4, c5, c6 = cumulants(6, dat2, 2)
  @test_approx_eq(convert(Array, c2),cn[1][fill(1:3, 2)...])
  @test_approx_eq(convert(Array, c3),cn[2][fill(1:3, 3)...])
  @test_approx_eq(convert(Array, c4),cn[3][fill(1:3, 4)...])
  @test_approx_eq(convert(Array, c5),cn[4][fill(1:3, 5)...])
  @test_approx_eq(convert(Array, c6),cn[5][fill(1:3, 6)...])


  # test higher cumulants for square last block
  c2, c3, c4, c5, c6, c7, c8 = cumulants(8, dat3, 2)
  cnn = snaivecumulant(dat3, 8)
  @test_approx_eq(convert(Array, c2),cnn["c2"])
  @test_approx_eq(convert(Array, c3),cnn["c3"])
  @test_approx_eq(convert(Array, c4),cnn["c4"])
  @test_approx_eq(convert(Array, c5),cnn["c5"])
  @test_approx_eq(convert(Array, c6),cnn["c6"])
  @test_approx_eq(convert(Array, c7),cnn["c7"])
  @test_approx_eq(convert(Array, c8),cnn["c8"])
   #@test_approx_eq(convert(Array, c9),c["c9"][fill(1:4, 9)...])

   #test of semi naive algorithm using fd
   #warnings in tests from forward diff, but it works
   # for julia type 5 forward diff does not work
  #  if VERSION < v"0.5.0-dev+1204"
  #    csm = snaivecumulant(dat3, 6)
  #    cfd = cumulantsfd(dat3, 6)
  #    @test_approx_eq(cfd[2-1],csm["c2"])
  #    @test_approx_eq(cfd[3-1],csm["c3"])
  #    @test_approx_eq(cfd[4-1],csm["c4"])
  #    @test_approx_eq(cfd[5-1],csm["c5"])
  #    @test_approx_eq(cfd[6-1],csm["c6"])
  #    #@test_approx_eq(cfd[7],csm["c7"])
  #  end

 export snaivecumulant, get_diff

end
