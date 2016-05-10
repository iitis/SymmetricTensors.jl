module Test
  using Base.Test
  using NullableArrays
  using SymmetricMatrix
  importall SymmetricMatrix
  
  function generatedata(n::Int = 20, segments::Int = 5, l::Int = 1000)
      randmatrix = randn(n,n)
      symmatrix = randmatrix*transpose(randmatrix)
      randmatrix, symmatrix, convert(BoxStructure{Float64}, symmatrix, segments), randn(l,n)
   end
   
   function createsegments(randmatrix, nonullel::Bool = false, s1::Int = 4)
      structure = NullableArray(Matrix{Float64}, s1, s1)
      for i = 1:s1, j = i:s1
	  structure[i,j] = randmatrix
      end
      if nonullel
	structure[2,1] = randmatrix
      end
      structure
  end
    
  
  function runtests()
    
    m, sm, smseg, data = generatedata()

    smseg1 = convert(BoxStructure{Float64}, sm, 2)

    @test_approx_eq(matricise(smseg), sm)
    @test_approx_eq(smseg*smseg, sm*sm)
    @test_approx_eq(matricise(smseg+smseg), sm+sm)
    @test_approx_eq(smseg*m, sm*m)
    @test_approx_eq(smseg*m[:,1:12], sm*m[:,1:12])
    @test_approx_eq(vec(smseg), vec(sm))
    @test_approx_eq(trace(smseg), trace(sm))
    @test_approx_eq(vecnorm(smseg), vecnorm(sm))
    @test_approx_eq(matricise(square(smseg)), sm*sm)
    @test_approx_eq(matricise(covbs(data, size(smseg.frame,1), false)), cov(data, corrected=false))
    @test_approx_eq(matricise(covbs(data, size(smseg.frame,1), true)), cov(data, corrected=true))
    

    
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, m, 5))
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, sm[:,1:15], 5))
    @test_throws(DimensionMismatch, convert(BoxStructure{Float64}, m, 7))
    @test_throws(DimensionMismatch, BoxStructure(smseg.frame[:,1:2]))
    @test_throws(DimensionMismatch, BoxStructure(createsegments(sm[:,1:2])))
    @test_throws(DimensionMismatch, BoxStructure(createsegments(m)))
    @test_throws(ArgumentError, BoxStructure(createsegments(sm, true)))
    @test_throws(DimensionMismatch, smseg*smseg1)
    @test_throws(DimensionMismatch, smseg+smseg1)
    @test_throws(DimensionMismatch, smseg*(m[1:5,:]))
    @test_throws(DimensionMismatch, smseg*(m[:,1:7]))
    @test_throws(DimensionMismatch, covbs(data, 7))
  end
    
    
  m, sm, smseg, data = generatedata()
  
    
  export runtests
end
