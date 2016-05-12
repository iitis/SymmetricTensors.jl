module SymmetricMatrix
using NullableArrays
import Base: trace, vec, vecnorm, +, *, .*, size

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
readsegments{T <: AbstractFloat}(i::Int, j::Int, data::BoxStructure{T}) = isnull(data.frame[i,j])? transpose(data.frame[j,i].value): data.frame[i,j].value
size{T <: AbstractFloat}(m1::BoxStructure{T}) = m1.sizesegment, size(m1.frame, 1), m1.sizesegment*size(m1.frame, 1)
segmentmult{T <: AbstractFloat}(k::Int, l::Int, m1::BoxStructure{T}...) = mapreduce(i -> readsegments(k,i, m1[1])*readsegments(i,l, m1[size(m1,1)]), +, collect(1:size(m1[1])[2]))


function testsize{T <: AbstractFloat}(m1::BoxStructure{T}...)
    for i = 2:size(m1,1)
        size(m1[1]) == size(m1[i]) || throw(DimensionMismatch("dims of B1 $(size(m1[1])) must equal to dims of B2 $(size(m1[i]))"))
    end
end

function bsoperation{T <: AbstractFloat}(bsfunction::Function, m1::BoxStructure{T}...)
    testsize(m1...)
    ret = zeros(T, size(m1[1])[3], size(m1[1])[3])
    for i = 1:size(m1[1])[2], j = 1:size(m1[1])[2]
        ret[seg(i, size(m1[1])[1]),seg(j, size(m1[1])[1])] = bsfunction(i,j, m1...)
    end
    ret
end


function oponbs{T <: AbstractFloat}(f::Function, m1::BoxStructure{T}...)
    ret = NullableArray(Matrix{T}, size(m1[1].frame))
      for k = 1:size(m1[1])[2], l = k:size(m1[1])[2]
        ret[k,l] = f(k,l, m1...)
      end
      BoxStructure(ret)
end

function blockop{T <: AbstractFloat, S <: Real}(n::S, f::Function, m1::BoxStructure{T}...)
    testsize(m1...)
    oponbs((i, j, m1...) -> f(n, map(k -> m1[k].frame[i,j].value,collect(1:size(m1,1)))...) ,m1...)
end

matricise{T <: AbstractFloat}(m1::BoxStructure{T}) = bsoperation(readsegments, m1)
*{T <: AbstractFloat}(m1::BoxStructure{T},  m2::BoxStructure{T}) = bsoperation(segmentmult, m1, m2)

*{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S) = blockop(n, *, m1)
+{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S) = blockop(n, +, m1)
+{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T}) = blockop(0, +, m1, m2)
.*{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T}) = blockop(1, (a::Int,b::Matrix{T},c::Matrix{T}) -> b.*c, m1, m2)
square{T <: AbstractFloat}(m1::BoxStructure{T}) = oponbs(segmentmult, m1)
trace{T <: AbstractFloat}(m1::BoxStructure{T}) = mapreduce(i -> trace(m1.frame[i,i].value), +, collect(1:size(m1)[2]))
vecnorm{T <: AbstractFloat}(m1::BoxStructure{T}) = sqrt(trace(square(m1)))

function vec{T <: AbstractFloat}(m1::BoxStructure{T})
    ret = T[]
    for k = 1:size(m1)[2], j = 1:size(m1)[1], i = 1:size(m1)[2]
        ret = (vcat(ret, readsegments(i,k, m1)[:,j]))
    end
    ret  
end

mattoslises{T <: AbstractFloat}(m2::Matrix{T}, slisesize::Int) = map(i -> m2[:,seg(i, slisesize)], collect(1:div(size(m2,2),slisesize)))

function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::Matrix{T})
      size(m1)[3] == size(m2,1) || throw(DimensionMismatch("size of B1 $(size(m1)[3]) must equal to size of A $(size(m2,1))"))
      size(m2,2)%size(m1)[1] == 0 || throw(DimensionMismatch(" matrix size $(size(m2,2)) / segment size $(size(m1)[1]) not integer"))
      m2slises = mattoslises(m2, size(m1)[1])
      ret = zeros(T, size(m1)[3], size(m2,2))
      for k1 = 1:size(m1)[2], k = 1:size(m2slises,1)
        ret[seg(k1, size(m1)[1]),seg(k, size(m1)[1])] = mapreduce(j -> readsegments(k1,j, m1)*m2slises[k][seg(j, size(m1)[1]),:], +, collect(1:size(m1)[2]))
      end
      ret
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


