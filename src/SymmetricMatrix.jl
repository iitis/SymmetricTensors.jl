module SymmetricMatrix
using NullableArrays
import Base: trace, vec, vecnorm, +, *

indexes(i, ofset) = ((i-1)*ofset+1):(i*ofset)

function segmentise{T <: AbstractFloat}(data::Matrix{T}, segments::Int)
    issym(Array{Float32}(data)) || throw(DimensionMismatch("matrix is not symmetric")) #poprawic
    (size(data,1)%segments == 0) || throw(DimensionMismatch("data size $(size(data,1)) / segment size $segments not integer"))
    frame = NullableArray(Matrix{T}, segments, segments)
    ofset = div(size(data,1), segments)
    for i = 1:segments, j = i:segments
        frame[i,j] = data[indexes(i, ofset), indexes(j, ofset)]
    end
    frame
end

function structfeatures{T <: AbstractFloat}(frame::NullableArrays.NullableArray{Matrix{T},2})
    sf = size(frame, 1)
    sf == size(frame, 2) || throw(DimensionMismatch("frame not symmetric"))
    for i = 1:sf, j = 1:sf
        if i > j
            isnull(frame[i,j]) || throw(ArgumentError("underdiagonal block [ $i , $j ] not null"))
        else    
            size(frame[i,j].value, 1) == size(frame[i,j].value, 2) || throw(DimensionMismatch("[ $i , $j ] block not squared"))
        end          
        issym(Array{Float32}(frame[i,i].value)) || throw(DimensionMismatch("diagonal blocks not symmetric")) #poprawic
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

getdims{T <: AbstractFloat}(m1::BoxStructure{T}) = m1.sizesegment, size(m1.frame, 1)

convert{T <: AbstractFloat}(::Type{BoxStructure{T}}, data::Matrix{T}, segments::Int = 2) = BoxStructure(segmentise(data, segments))

readsegments{T <: AbstractFloat}(data::BoxStructure{T},  i::Int, j::Int) = isnull(data.frame[i,j])? transpose(data.frame[j,i].value): data.frame[i,j].value


function segmentmult{T <: AbstractFloat}(k::Int, l::Int, m1::BoxStructure{T}, m2::BoxStructure{T}, s1::Int, s2::Int, blocknumber::Int)
      res = zeros(T, s1, s2)
	  for i = 1:blocknumber
	      res += readsegments(m1, k,i)*readsegments(m2, i,l)
	  end
    return res
end

function makematrix(T::Type, ofset::Int, s::Int)
    msize = s*ofset
    msize, zeros(T, msize,msize)
end

function matricise{T <: AbstractFloat}(m1::BoxStructure{T})
    ofset, s = getdims(m1)
    msize, matrix = makematrix(T, ofset, s)
    for i = 1:s, j = 1:s
        matrix[indexes(i, ofset),indexes(j, ofset)] = readsegments(m1,i,j)
    end
    matrix
end

function trace{T <: AbstractFloat}(m1::BoxStructure{T})
    tr = 0
    for i = 1:size(m1.frame, 1)
        tr += trace(m1.frame[i,i].value)
    end
      tr
end

function vec{T <: AbstractFloat}(m1::BoxStructure{T})
    ofset, s = getdims(m1)
    v = T[]
    for k = 1:s, j = 1:ofset, i = 1:s
        v = (vcat(v, readsegments(m1 ,i, k)[:,j]))
    end
      v  
end

function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T})
    ofset, s = getdims(m1)
    getdims(m1) == getdims(m2) || throw(DimensionMismatch("dims of B1 $(getdims(m1)) must equal to dims of B2 $(getdims(m2))"))
    msize, matrix = makematrix(T, ofset, s)
    for k = 1:s, l = 1:s
        matrix[indexes(k, ofset),indexes(l, ofset)] = segmentmult(k,l, m1, m2, ofset, ofset, s)
    end
      matrix
end
    
function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::Matrix{T})
      ofset, s1 = getdims(m1)
      msize, matrix = makematrix(T, ofset, s1)
      msize == size(m2,1) || throw(DimensionMismatch("size of B1 $(msize) must equal to size of A $(size(m2,1))"))
      size(m2,2)%ofset == 0 || throw(DimensionMismatch(" matrix size $(size(m2,2)) / segment size $ofset not integer"))
      s2 = div(size(m2,2),ofset)
      arraysegments = cell(s2)
      matrix = matrix[:, 1:size(m2,2)]
      for i = 1:s2
        arraysegments[i] = m2[:,indexes(i, ofset)]
      end
      for k1 = 1:s1, k = 1:s2
        a = zeros(ofset, ofset)
        for j = 1:s1
            a += readsegments(m1,k1,j)*arraysegments[k][indexes(j, ofset),:]
        end
        matrix[indexes(k1, ofset),indexes(k, ofset)] = a
      end
      matrix
  end

function square{T <: AbstractFloat}(m1::BoxStructure{T})
    ofset, s = getdims(m1)
    blockstruct = NullableArray(Matrix{T}, s, s)
      for k = 1:s, l = k:s
        blockstruct[k,l] = segmentmult(k,l, m1, m1, ofset, ofset, s)
      end
      BoxStructure(blockstruct)
  end

vecnorm{T <: AbstractFloat}(m1::BoxStructure{T}) = sqrt(trace(square(m1)))

function +{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T})
    s = size(m1.frame, 1)
    getdims(m1) == getdims(m2) || throw(DimensionMismatch("dims of B1 $(getdims(m1)) must equal to dims of B2 $(getdims(m2))"))
    res = NullableArray(Matrix{T}, s, s)
    for i = 1:s, j = 1:s
        if !isnull(m1.frame[i,j])
            res[i,j] = m1.frame[i,j].value+m2.frame[i,j].value
        end
    end
    BoxStructure(res) 
end

function covbs{T <: AbstractFloat}(datatab::Matrix{T}, blocksize::Int = 2, corrected::Bool = false)
    size(datatab, 2)%blocksize == 0 || throw(DimensionMismatch("data size $(size(datatab, 2)) / segment size $blocksize not integer"))
    s = div(size(datatab, 2), blocksize)   
    cmatrix = NullableArray(Matrix{T}, s, s)
    for b1 = 1:s, b2 = b1:s
        cmatrix[b1,b2] = cov(datatab[:,indexes(b1, blocksize)], datatab[:,indexes(b2, blocksize)], corrected = corrected)
    end
    BoxStructure(cmatrix)
end
  
export BoxStructure, convert, matricise, trace, vec, *, square, vecnorm, +, covbs
end

# dokladnosci przy dodawaniu


