module Test
  using Base.Test
  using NullableArrays


  function generatesym(n::Int)
      A = randn(n,n)
      return A * A'
  end

    function operationtesting(testoperator::Function, compareoperator::Function, segmentingf::Function, numbervar::Int = 2
    , numbsegmentedvar::Int = 2, segsize::Int = 10, segn::Int = 10)
      variable = cell(numbervar)
      segmentedvar = cell(numbervar)
      for i = 1:numbervar
	if i <= numbsegmentedvar
	  variable[i] = generatesym(segsize*segn) 
	  segmentedvar[i] = segmentingf(variable[i], segn)
	else
	  segmentedvar[i] = variable[i] = randn(segsize*segn, segsize*segn)
	end
      end
      @test_approx_eq(testoperator(segmentedvar...), compareoperator(variable...))
    end
    
   function operationtesting(testoperator::Function, compareoperator::Function, segmentingf::Function, desegmentingf::Function, numbervar::Int = 2
    , numbsegmentedvar::Int = 2, segsize::Int = 10, segn::Int = 10)
      variable = cell(numbervar)
      segmentedvar = cell(numbervar)
      for i = 1:numbervar
	if i <= numbsegmentedvar
	  variable[i] = generatesym(segsize*segn) 
	  segmentedvar[i] = segmentingf(variable[i], segn)
	else
	  segmentedvar[i] = variable[i] = randn(segsize*segn, segsize*segn)
	end
      end
      println(typeof(desegmentingf(testoperator(segmentedvar...))), typeof(compareoperator(variable...)))
      @test_approx_eq(desegmentingf(testoperator(segmentedvar...)), compareoperator(variable...))
    end


    function dataoperatortest(testdop::Function, desegmentingf::Function, segsize::Int = 4, segn::Int = 4, l::Int = 100)
      data = randn(l, segsize*segn)
      @test_approx_eq(desegmentingf(testdop(data, segn)), cov(data, corrected = false)) #o tym pomyslec
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
    
  export operationtesting, dataoperatortest, teststructure, nonsquaredframe, nonullframe, notsymdiagblocks
end
