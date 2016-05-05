module Test
  using Base.Test


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
testdop
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


    function dataoperatortest(testdop::Function, comparedop::Function, desegmentingf::Function, segsize::Int = 4, segn::Int = 4, l::Int = 100)
      data = randn(l, segsize*segn)
      @test_approx_eq(desegmentingf(testdop(data, segn)), comparedop(data, corrected = false)) #o tym pomyslec
    end
    
    
  export operationtesting, dataoperatortest
end
