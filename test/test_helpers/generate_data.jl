srand(42)

function symmetrise(m::Array)
  ret = zeros(m)
  indices = product(fill(collect(1:size(m, 1)), ndims(m))...)
  r = 1
  for i in indices
    if issorted(i)
      for k in collect(permutations(i))
        ret[k...] = m[r]
      end
      r += 1
    end
  end
  ret
end

function generatedata(n::Int = 15, seg::Int = 4, l::Int = 1000)
    rmat = randn(n,n,n)
    srmat = symmetrise(rmat)
    rmat, srmat, convert(SymmetricTensor, srmat, seg)
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

function clcopulagen(t::Int, m::Int)
    theta = 1.02
    coredist = Gamma(1,1/theta)
    x = rand(t)
    u = rand(t,m)
    ret = zeros(t,m)
    invphi(x::Array{Float64,1}, theta::Float64) = (1+ theta.*x).^(-1/theta)
    for i = 1:m
        uniform = invphi(-log(u[:,i])./quantile(coredist, x), theta)
        ret[:,i] = quantile(Weibull(1+0.01*i,1), uniform)
    end
    ret
end
