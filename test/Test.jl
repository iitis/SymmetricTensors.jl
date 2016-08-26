module Test
  using Base.Test
  using NullableArrays
  using Boxtensors
  using Iterators
  using Tensors
  using Distributions
  using ForwardDiff
  importall Boxtensors

  symmetrise{T <: AbstractFloat}(matrix::Matrix{T}) = matrix*transpose(matrix)

  function get_diff{T<:AbstractFloat}(dane::Matrix{T}, r::Int = 4)
    n = size(dane, 2)
    f(t::Vector) = log(mean(exp(t'*dane')))
    t_vec = zeros(Float64, n)
    fgen1 = ForwardDiff.gradient(f)
    fgen2 = hessian(f)
    fgen3 = tensor(f)
    function nthcumgen(gen_funct)
        vecform(x) = vec(gen_funct(x::Vector))
        jacobian(vecform)
    end
    tensor_form(mat::Matrix, size::Int, modes::Int) = reshape(mat,fill(size,modes)...)
    fgen = fgen3
    ret = Any[]
    for modes = 4:r
        fgen = nthcumgen(fgen)
        fn = tensor_form(fgen(t_vec),n, modes)
        push!(ret, fn)
    end
    fgen1(t_vec), fgen2(t_vec), fgen3(t_vec), ret...
  end

  function permute(array::Vector{Int})
      a = Vector{Vector{Int}}[]
      n = size(array, 1)
      for p in partitions(array)
          add = true
          for k in p
              if size(k,1) in[1, n]
                  add = false
              end
          end
          if add
              push!(a, p)
          end
      end
      return a
  end

  macro per(a, b ,i)
      eval = quote
          if size($i,1) == 3
              I = [$i[1], $i[2], $i[3]]
          elseif size($i,1) == 4
              I = [$i[1], $i[2], $i[3], $i[4]]
          elseif size($i,1) == 5
              I = [$i[1], $i[2], $i[3], $i[4], $i[5]]
          elseif size($i,1) == 6
              I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6]]
          elseif size($i,1) == 7
              I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7]]
          elseif size($i,1) == 8
              I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7], $i[8]]
          elseif size($i,1) == 9
              I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7], $i[8], $i[9]]
          elseif size($i,1) == 10
              I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7], $i[8], $i[9], $i[10]]
          end
          for per in collect(permutations(I))
              $a[per...] = $b
          end
      end
      return (eval)
      end


  function permutations_output!{T<:AbstractFloat, N}(m4::AbstractArray{T, N}, a::T, list::Vector{Int})
      @per(m4, a, list)
  end


  function moment_element!{T<:AbstractFloat, N}(moment::Array{T, N}, indices::Vector{Int}, data::Matrix{T})
      multiple = 1
      for i in indices
          multiple  = multiple.*data[:,i]
      end
      permutations_output!(moment, mean(multiple), indices)
  end

  function moment3{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,3)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m
          moment_element!(moment, [i1,i2,i3], data)
      end
      return Array(moment)
  end


  function moment4{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,4)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m
          moment_element!(moment, [i1,i2,i3,i4], data)
      end
      return Array(moment)
  end


  function moment5{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,5)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m
          moment_element!(moment, [i1,i2,i3,i4,i5], data)
      end
      return Array(moment)
  end


  function moment6{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,6)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m
          moment_element!(moment, [i1,i2,i3,i4,i5,i6], data)
      end
      return Array(moment)
  end

  function moment7{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,7)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m
          moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7], data)
      end
      return Array(moment)
  end

  function moment8{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,8)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m
          moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7,i8], data)
      end
      return Array(moment)
  end


  function moment9{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,9)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m
          moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7,i8,i9], data)
      end
      return Array(moment)
  end

  function moment10{T<:AbstractFloat}(data::Matrix{T})
      m = size(data,2)
      moment = zeros(fill(m,10)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m, i10 = i9:m
          moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10], data)
      end
      return Array(moment)
  end

  function moment_n{T<:AbstractFloat}(data::Matrix{T}, n::Int)
      m = size(data,2)
      moment = zeros(fill(m,n)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m, i10 = i9:m
          indices = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10][1:n]
          moment_element!(moment, indices, data)
      end
      return Array(moment)
  end


  function calculate_el(c::Dict{ASCIIString,Any}, list::Vector{Int})
      a = permute(list)
      w = 0
      for k = 1:size(a,1)
          r = 1
          for el in a[k]
              r*= c["c"*string(size(el,1))][el...]
          end
          w += r
      end
      return w
  end

  function product4(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,4)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m
          indices = [i1,i2,i3,i4]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function product5(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,5)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m
          indices = [i1,i2,i3,i4, i5]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function product6(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,6)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m
          indices = [i1,i2,i3,i4, i5, i6]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function product7(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,7)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m
          indices = [i1,i2,i3,i4, i5, i6, i7]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function product8(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,8)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m
          indices = [i1,i2,i3,i4, i5, i6, i7, i8]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function product9(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,9)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m
          indices = [i1,i2,i3,i4, i5, i6, i7, i8, i9]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function product10(c::Dict{ASCIIString,Any})
      m = size(c["c2"], 2)
      m4 = zeros(fill(m,10)...)
      for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m, i10 = i9:m
          indices = [i1,i2,i3,i4, i5, i6, i7, i8, i9, i10]
          a = calculate_el(c, indices)
          permutations_output!(m4, a, indices)
      end
      return Array(m4)
  end

  function snaivecumulant{T<:AbstractFloat}(data::Matrix{T}, n::Int)
      data = centre(data);
      c2 = cov(data, corrected = false)
      c3 = moment3(data)
      cumulants = Dict("c2" => c2, "c3" => c3);
      if n == 3
          return cumulants
      end
      c4 = moment4(data) - product4(cumulants)
      cumulants = merge(cumulants, Dict("c4" => c4));
      if n == 4
          return cumulants
      end
      c5 = moment5(data)-product5(cumulants)
      cumulants = merge(cumulants, Dict("c5" => c5))
      if n == 5
          return cumulants
      end
      c6 = moment6(data)-product6(cumulants)
      cumulants = merge(cumulants, Dict("c6" => c6))
      if n == 6
          return cumulants
      end
      c7 = moment7(data)-product7(cumulants)
      cumulants = merge(cumulants, Dict("c7" => c7))
      if n == 7
          return cumulants
      end
      c8 = moment8(data)-product8(cumulants)
      cumulants = merge(cumulants, Dict("c8" => c8))
      if n == 8
          return cumulants
      end
      c9 = moment9(data)-product9(cumulants)
      cumulants = merge(cumulants, Dict("c9" => c9))
      if n == 9
          return cumulants
      end
      c10 = moment10(data)-product10(cumulants)
      cumulants = merge(cumulants, Dict("c10" => c10))
      return cumulants
  end


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
  @test_approx_eq(convert(Array, momentbs(dat, 3, 2)), moment3(dat))
  @test_approx_eq(convert(Array, momentbs(dat, 4, 2)), moment4(dat))

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

 export snaivecumulant, get_diff

end
