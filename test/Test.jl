module Test
  using Base.Test
  using NullableArrays
  using Boxtensors
  using Iterators
  using Tensors
  using Distributions
  importall Boxtensors

  symmetrise{T <: AbstractFloat}(matrix::Matrix{T}) = matrix*transpose(matrix)


  rmat = symmetrise(randn(6,6))
  converttest = convert(BoxStructure, rmat)
  @test_approx_eq(converttest.frame[1,1].value, rmat[1:3, 1:3])
  @test_approx_eq(converttest.frame[1,2].value, rmat[1:3, 4:6])
  @test_approx_eq(converttest.frame[2,2].value, rmat[4:6, 4:6])
  @test(isnull(converttest.frame[2,1]))


  function generatedata(seed::Int = 1234, n::Int = 15, seg::Int = 4, l::Int = 1000)
      srand(seed)
      rmat = randn(n,n)
      rmat, symmetrise(rmat), convert(BoxStructure, symmetrise(rmat), seg), rand(l,n), bitrand(n,n), (im*rmat + rmat)*(im*rmat + rmat)'
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
    smseg1 = convert(BoxStructure, sm, 2)
    m2, sm2, smseg2 = generatedata(1233)
    badsegments = 8

    @test_approx_eq(convert(Array, smseg), sm)
    @test_approx_eq(convert(Array,convert(BoxStructure, Matrix{Float16}(sm), 5)), Matrix{Float16}(sm))
    @test_approx_eq(convert(Array,convert(BoxStructure, Matrix{Float32}(sm), 5)), Matrix{Float32}(sm))
    @test_approx_eq(convert(Array,convert(BoxStructure, Matrix{AbstractFloat}(sm), 5)), Matrix{AbstractFloat}(sm))

    @test_approx_eq(smseg*smseg2, sm*sm2)
    @test_approx_eq(convert(Array,smseg+smseg2), sm+sm2)
    @test_approx_eq(convert(Array,smseg-smseg2), sm-sm2)
    @test_approx_eq(convert(Array,smseg.*smseg2), sm.*sm2)
    @test_approx_eq(convert(Array,smseg./smseg2), sm./sm2)
    @test_approx_eq(smseg*m, sm*m)
    @test_approx_eq(convert(Array,smseg*2.1), sm*2.1)
    @test_approx_eq(convert(Array,smseg/2.1), sm/2.1)
    @test_approx_eq(convert(Array,smseg*2), sm*2)
    @test_approx_eq(convert(Array,smseg+2.1), sm+2.1)
    @test_approx_eq(convert(Array,smseg-2.1), sm-2.1)
    @test_approx_eq(convert(Array,smseg+2), sm+2)
    @test_approx_eq(convert(Array, bcss(smseg, m)), m'*(smseg*m))
    @test_approx_eq(convert(Array, bcss(smseg, m[:,1:12])), (m[:,1:12])'*(smseg*m[:,1:12]))

    m4, sm4, smseg4 = generatedata()
    add(smseg4, 2.1)
    @test_approx_eq(convert(Array,smseg4), sm4+2.1)

    @test_approx_eq(smseg*m[:,1:12], sm*m[:,1:12])
    @test_approx_eq(smseg*m[:,1:7], sm*m[:,1:7])
    @test_approx_eq(smseg*m[:,1:1], sm*m[:,1:1])
    @test_approx_eq(vec(smseg), vec(sm))
    @test_approx_eq(trace(smseg), trace(sm))
    @test_approx_eq(vecnorm(smseg), vecnorm(sm))
    @test_approx_eq(convert(Array,square(smseg)), sm*sm)
    @test_approx_eq(convert(Array,covbs(data, smseg.sizesegment, false)), cov(data, corrected=false))
    @test_approx_eq(convert(Array, covbs(Matrix{Float32}(data), smseg.sizesegment, false)), cov(Matrix{Float32}(data), corrected=false))
    @test_approx_eq(convert(Array,covbs(data, smseg.sizesegment, true)), cov(data, corrected=true))
    
    @test_throws(AssertionError, convert(BoxStructure, m, 5))
    @test_throws(DimensionMismatch, convert(BoxStructure, sm[:,1:13], 6))
    @test_throws(DimensionMismatch, convert(BoxStructure, sm,  badsegments))
    @test_throws(MethodError, convert(BoxStructure, boolean, 5))
    @test_throws(MethodError, convert(BoxStructure, comlx, 5))
    @test_throws(AssertionError, BoxStructure(smseg.frame[:,1:2]))
    @test_throws(AssertionError, BoxStructure(createsegments(sm, false, 4, true)))
    @test_throws(AssertionError, BoxStructure(createsegments(m)))
    @test_throws(AssertionError, BoxStructure(createsegments(sm, true)))
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
    bstensor = (convert(BoxStructure, stensor, 3))
    stensor1 = genstensor(Float64, 4,10, 1233)
    bstensor1 = (convert(BoxStructure, stensor1, 3))
    stensor2 = genstensor(Float64, 4,11)
    bstensor2 = (convert(BoxStructure, stensor2, 3))
    stensor3 = genstensor(Float64, 3,10)
    bstensor3 = (convert(BoxStructure, stensor2, 3))

    # tests for tensors

    @test_approx_eq(convert(Array, (bstensor)), stensor)
    @test_approx_eq(convert(Array,convert(BoxStructure, Array{Float16}(stensor), 3)), Array{Float16}(stensor))
    @test_approx_eq(convert(Array,convert(BoxStructure, Array{Float32}(stensor), 3)), Array{Float32}(stensor))
    @test_approx_eq(convert(Array,convert(BoxStructure, Array{AbstractFloat}(stensor), 3)), Array{AbstractFloat}(stensor))

    @test_approx_eq(convert(Array,bstensor*2.1), stensor*2.1)
    @test_approx_eq(convert(Array,bstensor/2.1), stensor/2.1)
    @test_approx_eq(convert(Array,bstensor*2), stensor*2)
    @test_approx_eq(convert(Array,bstensor+2.1), stensor+2.1)
    @test_approx_eq(convert(Array,bstensor-2.1), stensor-2.1)
    @test_approx_eq(convert(Array,bstensor+2), stensor+2)
    @test_approx_eq(vec(bstensor), vec(stensor))
    @test_approx_eq(convert(Array,bstensor+bstensor1), stensor+stensor1)
    @test_approx_eq(convert(Array,bstensor-bstensor1), stensor-stensor1)
    @test_approx_eq(convert(Array,bstensor.*bstensor1), stensor.*stensor1)
    @test_approx_eq(convert(Array,bstensor./bstensor1), stensor./stensor1)
    @test_approx_eq(modemult(bstensor, m[:,1:size(stensor, 1)], 1), Tensors.modemult(stensor, m[:,1:size(stensor, 1)], 1))
    @test_approx_eq(modemult(bstensor, m[:,1:size(stensor, 1)], 2), Tensors.modemult(stensor, m[:,1:size(stensor, 1)], 2))
    @test_approx_eq(modemult(bstensor, m[:,1:size(stensor, 1)], 4), Tensors.modemult(stensor, m[:,1:size(stensor, 1)], 4))

    @test_throws(DimensionMismatch, modemult(bstensor, m[:,1:2], 4))
    @test_throws(BoundsError, modemult(bstensor, m[:,1:size(stensor, 1)], ndims(stensor)+1))
    @test_throws(DimensionMismatch, bstensor+bstensor2)
    @test_throws(DimensionMismatch, bstensor.*bstensor2)
    @test_throws(DimensionMismatch, bstensor+bstensor3)
    @test_throws(DimensionMismatch, bstensor.*bstensor3)
    @test_throws(AssertionError, BoxStructure(bstensor.frame[:,:,:,1:2]))
    @test_throws(AssertionError, BoxStructure(createsegments(randtensor(Float64, 4, 10))))
    @test_throws(AssertionError, BoxStructure(createsegments(sm, false, 4, true)))
    @test_throws(AssertionError, BoxStructure(createsegments(stensor, true)))
    @test_throws(DimensionMismatch, convert(BoxStructure, stensor,  badsegments))
    @test_throws(MethodError, convert(BoxStructure, randtensor(Bool, 4, 10), 3))
    @test_throws(MethodError, convert(BoxStructure, randtensor(Complex64, 4, 10), 3))


    function multimodemult{T <: AbstractFloat, N}(t::Array{T,N}, mat::Matrix{T})
      ret = t
      for i = 1:N
	  ret = Tensors.modemult(ret, mat, i)
      end
      ret
    end

    @test_approx_eq(convert(Array, bcssclass(bstensor, m[1:6, 1:10], 3)), multimodemult(stensor, m[1:6, 1:10]))
    @test_approx_eq(convert(Array, bcssclass(bstensor, m[1:7, 1:10], 3)), multimodemult(stensor, m[1:7, 1:10]))
    @test_approx_eq_eps(sum(abs(mean(centre(m[1:3, 1:10]), 1))), 0, 1e-15)

  #rests moments via semi naive algorithms
  dat = centre(data[1:15,1:5])
  @test_approx_eq(convert(Array, momentbc(dat, 3, 2)), moment3(dat))
  @test_approx_eq(convert(Array, momentbc(dat, 4, 2)), moment4(dat))

  # kopula Claytona rozklady brzegowe Weibulla
  function clcopulatest(t::Int, m::Int)
    theta = 1.02
    coredist = Gamma(1,1/theta)
    srand(1256)
    x = rand(t)
    srand(1235)
    u = rand(t,m)
    ret = zeros(t,m)
    invphi(x::Array{Float64,1}, theta::Float64) = (1+ theta.*x).^(-1/theta)
    for i = 1:m
        uniform = invphi(-log(u[:,i])./quantile(coredist, x), theta)
        ret[:,i] = quantile(Weibull(1+0.01*i,1), uniform)
    end
    ret
  end

  dat1 = clcopulatest(10, 5)
  dat2 = dat1[:,1:4]
  dat3 = dat1[:,1:2]

  #test of semi naive algorithm using fd
  #warnings in tests from forward diff, but it works
  csm = snaivecumulant(dat3, 6)
  cfd = get_diff(dat3, 6)
  @test_approx_eq(cfd[2],csm["c2"])
  @test_approx_eq(cfd[3],csm["c3"])
  @test_approx_eq(cfd[4],csm["c4"])
  @test_approx_eq(cfd[5],csm["c5"])
  @test_approx_eq(cfd[6],csm["c6"])
  #@test_approx_eq(cfd[7],csm["c7"])

  # test the bs algorithm using the semi naive (for non square last block)
  c = snaivecumulant(dat1, 6)
  c2, c3, c4, c5, c6 = cumulants(6, dat1, 2)
  @test_approx_eq(convert(Array, c2),c["c2"])
  @test_approx_eq(convert(Array, c3),c["c3"])
  @test_approx_eq(convert(Array, c4),c["c4"])
  @test_approx_eq(convert(Array, c5),c["c5"])
  @test_approx_eq(convert(Array, c6),c["c6"])
 # @test_approx_eq(convert(Array, c7),c["c7"])

 # for square last block
  c2, c3, c4, c5, c6 = cumulants(6, dat2, 2)
  @test_approx_eq(convert(Array, c2),c["c2"][fill(1:4, 2)...])
  @test_approx_eq(convert(Array, c3),c["c3"][fill(1:4, 3)...])
  @test_approx_eq(convert(Array, c4),c["c4"][fill(1:4, 4)...])
  @test_approx_eq(convert(Array, c5),c["c5"][fill(1:4, 5)...])
  @test_approx_eq(convert(Array, c6),c["c6"][fill(1:4, 6)...])
 # @test_approx_eq(convert(Array, c7),c["c7"][fill(1:4, 7)...])

end
