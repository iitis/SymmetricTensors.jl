module SymmetricMatrix

  squared(m::Array{Any}) = size(m,1) == size(m,2)? true: error("not squared")


  function test_frame(m::Array{Any})
      squared(m)
      for i = 1:size(m,1), j = i:size(m,2)
	  segment = m[i,j]
	  (eltype(segment) <: AbstractFloat)? true: error("wrong data type")
	  size(segment,1) == size(segment,2)? true: error("element not squared")
	  issym(Array{Float32}(m[i,i]))? true: error("diagonal element not symetric")
      end
      for i = 1:size(m,1), j = 1:(i-1)
	  try m[i,j]
	      return false
	  catch
	      return true
	  end
      end
  end

  immutable BoxStructure
      frame::Array{Any,2}
      sizeframe::Int
      sizesegment::Int
      BoxStructure(frame) = test_frame(frame)? new(frame, size(frame,1), size(frame[1,1],1)): error("box structure error")
  end

  function test_index(a, indices)
      for k in indices
	  k = k > size(a,1)? error("index out of range") : k
      end
  end


  function read_segments(data::Array,  i::Int, j::Int)
      test_index(data, [i,j])
      s = zeros(size(data[1,1]))
      try 
	  s = data[i,j]
      catch
	  s = data[j,i]'
      end
      s
  end


  function into_segments{T <: AbstractFloat}(matrix::Array{T,2}, segment_numb::Int)
      issym(matrix)? true: error("imput matrix not symetric")
      s = size(matrix,1)
      (s%segment_numb == 0)? true: error("wrong number of segments")
      blockstruct = cell(segment_numb, segment_numb)
      ofset = Int(s/segment_numb)
      for i = 1:segment_numb, j = i:segment_numb
	  blockstruct[i,j] = matrix[1+ofset*(i-1):ofset*i, 1+ofset*(j-1):ofset*j]
      end
      BoxStructure(blockstruct)
  end

  function segmentmult(k::Int, l::Int, m1::Array{Any,2}, m2::Array, s1::Int, s2::Int, blocknumber::Int)
      res = zeros(s1, s2)
	  for i = 1:blocknumber
	      res += read_segments(m1, k,i)*read_segments(m2, i,l)
	  end
      return res
  end


  function multiplebs(m1::BoxStructure)
      s = m1.sizeframe
      blockstruct = cell(s,s)
      for k = 1:s, l = k:s
	  blockstruct[k,l] = segmentmult(k,l, m1.frame, m1.frame, m1.sizesegment, m1.sizesegment, s)
      end
      BoxStructure(blockstruct)
  end


  function multiplebs(m1::BoxStructure, m2::BoxStructure)
      s = m1.sizeframe
      ofset = m1.sizesegment
      (s == m2.sizeframe)? true: error("different numbers of blocks")
      (ofset == m2.sizesegment)? true: error("different size of blocks")
      msize = s*ofset
      matrix = zeros(msize, msize)
      for k = 1:s, l = 1:s
	  matrix[((k-1)*ofset+1):(k*ofset),((l-1)*ofset+1):(l*ofset)] = 
	  segmentmult(k,l, m1.frame, m2.frame, ofset, ofset, s)
      end
      matrix
  end


  function multiplebs(m1::BoxStructure, m2::Array)
      ofset = m1.sizesegment
      s1 = m1.sizeframe
      (s1*ofset == size(m2,1))? true: error("dimentions...")
      arraysegments = cell(s1)   
      msize = s1*ofset
      matrix = zeros(size(m2))
      for i = 1:s1
	  arraysegments[i] = m2[((i-1)*ofset+1):(i*ofset),:]
      end
      for k = 1:s1
	  matrix[((k-1)*ofset+1):(k*ofset),:] = 
	  segmentmult(k,1, m1.frame, arraysegments, ofset, size(m2,2), s1)
      end
      matrix
  end


  function bstomatrix(m1::BoxStructure)
      ofset = m1.sizesegment
      msize = m1.sizeframe*ofset
      matrix = zeros(msize,msize)
      for i = 1:m1.sizeframe, j = 1:m1.sizeframe
	  matrix[((i-1)*ofset+1):(i*ofset),((j-1)*ofset+1):(j*ofset)] = read_segments(m1.frame,i,j)
      end
      matrix
  end

function add(m1::BoxStructure, m2::BoxStructure)
    s = m1.sizeframe
    (s == m2.sizeframe)? true: error("different numbers of blocks")
    (m1.sizesegment == m2.sizesegment)? true: error("different size of blocks")
    res = cell(s,s)
    for i = 1:s, j = 1:s
        try
            res[i,j] = m1.frame[i,j]+m2.frame[i,j]
        catch
            ()
        end
    end
    BoxStructure(res) 
end
      
  function vectorisebs(m1::BoxStructure)
      T = eltype(m1.frame[1,1])
      ofset = m1.sizesegment
      s = m1.sizeframe
      v = T[]
      for k = 1:s, j = 1:ofset, i = 1:s
	  v = (vcat(v, read_segments(m1.frame ,i, k)[:,j]))
      end
      v  
  end

  function tracebs(m1::BoxStructure)
      tr = 0
      for i = 1:m1.sizeframe
	  tr += trace(m1.frame[i,i])
      end
      tr
  end

  fnorm(m1::BoxStructure) = sqrt(tracebs(multiplebs(m1)))


  function generatesmat(n::Int)
      A = randn(n,n)
      return A * A'
  end

  
function covariancebs(datatab::Array{Float64, 2}, blocksize::Int)
    d = size(datatab, 2)
    (d%blocksize == 0)? true: error("wrong number of blocks")
    segmumber = Int(d/blocksize)   
    cmatrix = cell(segmumber,segmumber) 
    for b1 = 1:segmumber, b2 = b1:segmumber
        cmatrix[b1,b2] = cov(datatab[:,blocksize*(b1-1)+1:blocksize*b1], datatab[:,blocksize*(b2-1)+1:blocksize*b2], corrected = false)
    end
    BoxStructure(cmatrix)
end  
  
export into_segments, multiplebs, bstomatrix, vectorisebs, tracebs, fnorm, generatesmat, covariancebs, add
end
