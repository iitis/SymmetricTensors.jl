module Test
  using Base.Test
  using NullableArrays
  using SymmetricMatrix
  importall SymmetricMatrix


  function runtests()
    n = 20
    l = 1000
    segments = 5

    testmatrix = randn(n,n)
    sm = testmatrix*transpose(testmatrix);
    smseg = convert(BoxStructure{Float64}, sm, segments);
    data = randn(l,n);


    @test_approx_eq(matricise(smseg), sm)
    @test_approx_eq(smseg*smseg, sm*sm)
    @test_approx_eq(matricise(smseg+smseg), sm+sm)
    @test_approx_eq(smseg*testmatrix, sm*testmatrix)
    @test_approx_eq(vec(smseg), vec(sm))
    @test_approx_eq(trace(smseg), trace(sm))
    @test_approx_eq(vecnorm(smseg), vecnorm(sm))
    @test_approx_eq(matricise(square(smseg)), sm*sm)
    @test_approx_eq(matricise(covbs(data, segments, false)), cov(data, corrected=false))
    @test_approx_eq(matricise(covbs(data, segments, true)), cov(data, corrected=true))
  end
    
    
    function nonsquaredframe(s1::Int, s2::Int, n::Int)
      frame = NullableArray(Matrix{Float64}, s1, s2)
      A = randn(n,n)
      X = A*transpose(A);
      for i = 1:s1, j = i:s2
	      frame[i,j] = X
      end
      frame
  end


function nonullframe(s1::Int, n::Int)
      frame = NullableArray(Matrix{Float64}, s1, s1)
      A = randn(n,n)
      X = A*transpose(A);
      for i = 1:s1, j = 1:s1
	      frame[i,j] = X
      end
      frame
  end

function notsymdiagblocks(s1::Int, n::Int)
      frame = NullableArray(Matrix{Float64}, s1, s1)
      X = randn(n,n)
      for i = 1:s1, j = i:s1
	      frame[i,j] = X
      end
      frame
  end
  
  function teststructure(struct, case::Int = 1)  
    if case == 1 
      X = nonsquaredframe(3,4,5)
    elseif case == 2
      X = nonullframe(3,4)
    elseif case == 3
      X = notsymdiagblocks(3,4)
    end
    @test_throws(struct(X))
  end
    
  export runtests, nonsquaredframe, nonullframe, notsymdiagblocks
end
