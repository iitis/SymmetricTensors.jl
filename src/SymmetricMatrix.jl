module SymmetricMatrix
using NullableArrays
import Base: trace, vec, vecnorm, +, *, size

seg(i, ofset) = ((i-1)*ofset+1):(i*ofset)

function segmentise{T <: AbstractFloat}(data::Matrix{T}, segments::Int)
    (maximum(abs(data - transpose(data))) < 1e-7) || throw(DimensionMismatch("matrix is not symmetric")) #poprawic
    (size(data,1)%segments == 0) || throw(DimensionMismatch("data size $(size(data,1)) / segment size $segments not integer"))
    ret = NullableArray(Matrix{T}, segments, segments)
    ofset = div(size(data,1), segments)
    for i = 1:segments, j = i:segments
        ret[i,j] = data[seg(i, ofset), seg(j, ofset)]
    end
    ret
end

function structfeatures{T <: AbstractFloat}(frame::NullableArrays.NullableArray{Matrix{T},2})
    size(frame, 1) == size(frame, 2) || throw(DimensionMismatch("frame not symmetric"))
    for i = 1:size(frame, 1), j = 1:size(frame, 1)
        if i > j
            isnull(frame[i,j]) || throw(ArgumentError("underdiagonal block [ $i , $j ] not null"))
        else    
            size(frame[i,j].value, 1) == size(frame[i,j].value, 2) || throw(DimensionMismatch("[ $i , $j ] block not squared"))
        end          
        (maximum(abs(frame[i,i].value - transpose(frame[i,i].value))) < 1e-7) || throw(DimensionMismatch("diagonal block $i not symmetric")) #poprawi
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

convert{T <: AbstractFloat}(::Type{BoxStructure{T}}, data::Matrix{T}, segments::Int = 2) = BoxStructure(segmentise(data, segments))

readsegments{T <: AbstractFloat}(data::BoxStructure{T},  i::Int, j::Int) = isnull(data.frame[i,j])? transpose(data.frame[j,i].value): data.frame[i,j].value

size{T <: AbstractFloat}(m1::BoxStructure{T}) = m1.sizesegment, size(m1.frame, 1), m1.sizesegment*size(m1.frame, 1)

function segmentmult{T <: AbstractFloat}(k::Int, l::Int, m1::BoxStructure{T}, m2::BoxStructure{T}, s1::Int, s2::Int, blocksn::Int)
     ret = zeros(T, s1, s2)
     for i = 1:blocksn
	ret += readsegments(m1, k,i)*readsegments(m2, i,l)
     end
     ret
end

function matricise{T <: AbstractFloat}(m1::BoxStructure{T})
    ret = zeros(T, size(m1)[3], size(m1)[3])
    for i = 1:size(m1)[2], j = 1:size(m1)[2]
        ret[seg(i, size(m1)[1]),seg(j, size(m1)[1])] = readsegments(m1,i,j)
    end
    ret
end

function trace{T <: AbstractFloat}(m1::BoxStructure{T})
    ret = T(0)
    for i = 1:size(m1)[2]
        ret += trace(m1.frame[i,i].value)
    end
    ret
end

function vec{T <: AbstractFloat}(m1::BoxStructure{T})
    ret = T[]
    for k = 1:size(m1)[2], j = 1:size(m1)[1], i = 1:size(m1)[2]
        ret = (vcat(ret, readsegments(m1 ,i, k)[:,j]))
    end
    ret  
end

function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T})
    size(m1) == size(m2) || throw(DimensionMismatch("dims of B1 $(size(m1)) must equal to dims of B2 $(size(m2))"))
    ret = zeros(T, size(m1)[3], size(m1)[3])
    for k = 1:size(m1)[2], l = 1:size(m1)[2]
        ret[seg(k, size(m1)[1]),seg(l, size(m1)[1])] = segmentmult(k,l, m1, m2, size(m1)[1], size(m1)[1], size(m1)[2])
    end
    ret
end

function *{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S)
    ret = NullableArray(Matrix{T}, size(m1.frame))
      for i = 1:size(m1)[2], j = i:size(m1)[2]
        ret[i,j] = (m1.frame[i,j].value)*n
      end
      BoxStructure(ret)
  end



function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::Matrix{T})
      ret = zeros(T, size(m1)[3], size(m1)[3])
      size(m1)[3] == size(m2,1) || throw(DimensionMismatch("size of B1 $(size(m1)[3]) must equal to size of A $(size(m2,1))"))
      size(m2,2)%size(m1)[1] == 0 || throw(DimensionMismatch(" matrix size $(size(m2,2)) / segment size $(size(m1)[1]) not integer"))
      s2 = div(size(m2,2),size(m1)[1])
      arraysegments = cell(s2)
      ret = ret[:, 1:size(m2,2)]
      for i = 1:s2
        arraysegments[i] = m2[:,seg(i, size(m1)[1])]
      end
      for k1 = 1:size(m1)[2], k = 1:s2
        temporary = zeros(size(m1)[1], size(m1)[1])
        for j = 1:size(m1)[2]
            temporary += readsegments(m1,k1,j)*arraysegments[k][seg(j, size(m1)[1]),:]
        end
        ret[seg(k1, size(m1)[1]),seg(k, size(m1)[1])] = temporary
      end
      ret
  end

function square{T <: AbstractFloat}(m1::BoxStructure{T})
    ret = NullableArray(Matrix{T}, size(m1.frame))
      for k = 1:size(m1)[2], l = k:size(m1)[2]
        ret[k,l] = segmentmult(k,l, m1, m1, size(m1)[1], size(m1)[1], size(m1)[2])
      end
      BoxStructure(ret)
  end

vecnorm{T <: AbstractFloat}(m1::BoxStructure{T}) = sqrt(trace(square(m1)))

function +{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T})
    size(m1) == size(m2) || throw(DimensionMismatch("dims of B1 $(size(m1)) must equal to dims of B2 $(size(m2))"))
    ret = NullableArray(Matrix{T}, size(m1.frame))
    for i = 1:size(m1)[2], j = 1:size(m1)[2]
        if !isnull(m1.frame[i,j])
            ret[i,j] = m1.frame[i,j].value+m2.frame[i,j].value
        end
    end
    BoxStructure(ret) 
end

function covbs{T <: AbstractFloat}(datatab::Matrix{T}, blocksize::Int = 2, corrected::Bool = false)
    size(datatab, 2)%blocksize == 0 || throw(DimensionMismatch("data size $(size(datatab, 2)) / segment size $blocksize not integer"))
    s = div(size(datatab, 2), blocksize)   
    ret = NullableArray(Matrix{T}, s, s)
    for b1 = 1:s, b2 = b1:s
        ret[b1,b2] = cov(datatab[:,seg(b1, blocksize)], datatab[:,seg(b2, blocksize)], corrected = corrected)
    end
    BoxStructure(ret)
end
  
export BoxStructure, convert, matricise, trace, vec, *, square, vecnorm, +, covbs, size
end

# dokladnosci przy dodawaniu


