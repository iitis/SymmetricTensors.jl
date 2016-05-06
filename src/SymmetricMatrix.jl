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


# w formie nullable ten kod dziala


using NullableArrays

function segmentise{T <: AbstractFloat}(data::Matrix{T}, segments::Int)
    issym(Array{Float32}(data)) ? (): throw("SymmetryError")
    datasize = size(data,1)
    (datasize%segments == 0)? () : throw("SegmentNumberError")
    frame = NullableArray(Matrix{T}, segments, segments)
    segmentsize = div(datasize, segments)
    for i = 1:segments, j = i:segments
        frame[i,j] = data[1+segmentsize*(i-1):segmentsize*i, 1+segmentsize*(j-1):segmentsize*j]
    end
    frame
end

function structfeatures{T <: AbstractFloat}(frame::NullableArrays.NullableArray{Matrix{T},2})
    sf = size(frame, 1)
    sf == size(frame, 2)? (): throw("FrameNotSquared")
    for i = 1:sf, j = 1:sf
        if i > j
            isnull(frame[i,j])? (): throw("UnderDiagonalNotNull")
        else    
            size(frame[i,j].value, 1) == size(frame[i,j].value, 2)? (): throw("BlocksNotSquared")
        end          
        issym(Array{Float32}(frame[i,i].value))? (): throw("DiagBlocksNotSymmetric")
    end
end

immutable BoxStructure{T <: AbstractFloat} 
    frame::NullableArrays.NullableArray{Matrix{T},2}
    sizesegment::Int
    function call{T}(::Type{BoxStructure}, frame::NullableArrays.NullableArray{Matrix{T},2})
        structfeatures(frame)
        new{T}(frame, size(frame[1,1].value,1))
    end
end

convert{T <: AbstractFloat}(::Type{BoxStructure}, data::Matrix{T}, segments::Int = 2) =
BoxStructure(segmentise(data, segments))

readsegments{T <: AbstractFloat}(data::BoxStructure{T},  i::Int, j::Int) =
isnull(data.frame[i,j])? transpose(data.frame[j,i].value): data.frame[i,j].value

function matricise(m1::BoxStructure)
    ofset = m1.sizesegment
    framesize = size(m1.frame, 1)
    msize = framesize*ofset
    matrix = zeros(msize,msize)
    for i = 1:framesize, j = 1:framesize
        matrix[((i-1)*ofset+1):(i*ofset),((j-1)*ofset+1):(j*ofset)] = readsegments(m1,i,j)
    end
    matrix
end


function trace(m1::BoxStructure)
    tr = 0
    for i = 1:size(m1.frame, 1)
        tr += Base.trace(m1.frame[i,i].value)
    end
      tr
end


function vec(m1::BoxStructure)
    T = eltype(m1.frame[1,1])
    ofset = m1.sizesegment
    s = size(m1.frame, 1)
    v = T[]
    for k = 1:s, j = 1:ofset, i = 1:s
        v = (vcat(v, readsegments(m1 ,i, k)[:,j]))
    end
      v  
end

function segmentmult(k::Int, l::Int, m1::BoxStructure, m2::BoxStructure, s1::Int, s2::Int, blocknumber::Int)
      res = zeros(s1, s2)
	  for i = 1:blocknumber
	      res += readsegments(m1, k,i)*readsegments(m2, i,l)
	  end
    return res
end

function multiple(m1::BoxStructure, m2::BoxStructure)
    s = size(m1.frame, 1)
    ofset = m1.sizesegment
    (s == size(m1.frame, 2))? () : throw("different numbers of blocks")
    (ofset == m2.sizesegment)? () : throw("different size of blocks")
    msize = s*ofset
    matrix = zeros(msize, msize)
    for k = 1:s, l = 1:s
        matrix[((k-1)*ofset+1):(k*ofset),((l-1)*ofset+1):(l*ofset)] = 
        segmentmult(k,l, m1, m2, ofset, ofset, s)
    end
      matrix
end

multiple(m1::BoxStructure) = multiple(m1::BoxStructure, m1::BoxStructure)

vecnorm(m1::BoxStructure) = sqrt(Base.trace(multiple(m1)))

function add{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T})
    s = size(m1.frame, 1)
    (s == size(m1.frame, 2))? (): throw("different numbers of blocks")
    (m1.sizesegment == m2.sizesegment)? (): throw("different size of blocks")
    res = NullableArray(Matrix{T}, s, s)
    for i = 1:s, j = 1:s
        if !isnull(m1.frame[i,j])
            res[i,j] = m1.frame[i,j].value+m2.frame[i,j].value
        end
    end
    BoxStructure(res) 
end


