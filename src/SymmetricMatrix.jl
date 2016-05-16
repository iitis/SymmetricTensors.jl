module SymmetricMatrix
using NullableArrays
import Base: trace, vec, vecnorm, +, *, .*, size

#seg(i::Int, ofset::Int) = ((i-1)*ofset+1):(i*ofset)

seg(i::Int, of::Int, limit::Int = 1000) = (i*of <= limit)? (((i-1)*of+1):(i*of)): (((i-1)*of+1):limit)

function creatnarray1{T <: AbstractFloat}(data::Matrix{T}, rule::Function, segments::Int)
    ret = NullableArray(Matrix{T}, segments, segments)
    for i = 1:segments, j = i:segments
        ret[i,j] = rule(i,j, data)::Matrix{T}
    end
    ret
end

function creatnarray{T <: AbstractFloat}(data::Matrix{T}, rule::Function, segments::Int)
    arraysize = (size(data,1)%segments == 0)? segments: segments+1
    ret = NullableArray(Matrix{T}, fill(arraysize, 2)...)
    for i = 1:arraysize, j = i:arraysize
        ret[i,j] = rule(i,j, data)::Matrix{T}
    end
    ret
end

function segmentise1{T <: AbstractFloat}(data::Matrix{T}, segments::Int)
    (maximum(abs(data - transpose(data))) < 1e-7) || throw(DimensionMismatch("matrix is not symmetric")) #poprawic
    (size(data,1)%segments == 0) || throw(DimensionMismatch("data size $(size(data,1)) / segment size $segments not integer"))
    creatnarray(data, (i::Int,j::Int, data::Matrix{T}) -> data[seg(i, size(data,1)÷segments), seg(j, size(data,1)÷segments)], segments)
end

function segmentise{T <: AbstractFloat}(data::Matrix{T}, segments::Int)
    (maximum(abs(data - transpose(data))) < 1e-7) || throw(DimensionMismatch("matrix is not symmetric")) #poprawic
    ((size(data,1)%segments) <= (size(data,1)÷segments)) || throw(DimensionMismatch("last segment len $(size(data,1)-segments*(size(data,1)÷segments)) > segment len $(size(data,1)÷segments)"))
    # 20÷9 = 2 to to samo co 20÷10
    creatnarray(data, (i::Int,j::Int, data::Matrix{T}) -> data[seg(i, size(data,1)÷segments, size(data, 1)), seg(j, size(data,1)÷segments, size(data, 1))], segments)
end

function structfeatures{T <: AbstractFloat}(frame::NullableArrays.NullableArray{Matrix{T},2})
    isequal(size(frame)...) || throw(DimensionMismatch("frame not symmetric"))
    for i = 1:size(frame, 1), j = 1:size(frame, 1)
	if (i > j)
	  isnull(frame[i,j]) || throw(ArgumentError("underdiagonal block [ $i , $j ] not null"))
	elseif (i < size(frame, 1) & j < size(frame, 1))
	  isequal(size(frame[i,j].value)...) || throw(DimensionMismatch("[ $i , $j ] block not squared"))     
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

segmentmult{T <: AbstractFloat}(k::Int, l::Int, m1::BoxStructure{T}...) = mapreduce(i -> readsegments(k,i, m1[1])*readsegments(i,l, m1[size(m1,1)]), +, collect(1:size(m1[1])[2]))


function testsize{T <: AbstractFloat}(m1::BoxStructure{T}...)
    for i = 2:size(m1,1)
        size(m1[1]) == size(m1[i]) || throw(DimensionMismatch("dims of B1 $(size(m1[1])) must equal to dims of B2 $(size(m1[i]))"))
    end
end

function bsoperation{T <: AbstractFloat}(bsfunction::Function, m1::BoxStructure{T}...)
    testsize(m1...)
    hcat(map(j -> vcat(map(i -> bsfunction(i,j, m1...)::Matrix{T}, collect(1:size(m1[1])[2]))...), collect(1:size(m1[1])[2]))...)
end

function operationonbs{T <: AbstractFloat}(f::Function, m1::BoxStructure{T}...)
    ret = NullableArray(Matrix{T}, size(m1[1].frame))
      for k = 1:size(m1[1])[2], l = k:size(m1[1])[2]
        ret[k,l] = f(k,l, m1...)::Matrix{T}
      end
      BoxStructure(ret)
end

function blockop{T <: AbstractFloat, S <: Real}(n::S, f::Function, m1::BoxStructure{T}...)
    testsize(m1...)
    operationonbs((i, j, m1...) -> f(n, map(k -> m1[k].frame[i,j].value,collect(1:size(m1,1)))...)::Matrix{T} ,m1...)
end

matricise{T <: AbstractFloat}(m1::BoxStructure{T}) = bsoperation(readsegments, m1)
*{T <: AbstractFloat}(m1::BoxStructure{T},  m2::BoxStructure{T}) = bsoperation(segmentmult, m1, m2)

*{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S) = blockop(n, *, m1)
+{T <: AbstractFloat, S <: Real}(m1::BoxStructure{T}, n::S) = blockop(n, +, m1)
+{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T}) = blockop(0, +, m1, m2)
.*{T <: AbstractFloat}(m1::BoxStructure{T}, m2::BoxStructure{T}) = blockop(1, (a::Int,b::Matrix{T},c::Matrix{T}) -> b.*c, m1, m2)
square{T <: AbstractFloat}(m1::BoxStructure{T}) = operationonbs(segmentmult, m1)
trace{T <: AbstractFloat}(m1::BoxStructure{T}) = mapreduce(i -> trace(m1.frame[i,i].value), +, collect(1:size(m1)[2]))
vecnorm{T <: AbstractFloat}(m1::BoxStructure{T}) = sqrt(trace(square(m1)))
vec{T <: AbstractFloat}(m1::BoxStructure{T}) = Base.vec(matricise(m1))

slisemat{T <: AbstractFloat}(m2::Matrix{T}, slisesize::Int) = map(i -> m2[:,seg(i, slisesize, size(m2, 2))], collect(1:ceil(Int, size(m2,2)/slisesize)))

size{T <: AbstractFloat}(m1::BoxStructure{T}) = m1.sizesegment, size(m1.frame, 1), m1.sizesegment*(size(m1.frame, 1)-1)+size(m1.frame[end,end].value,1)

function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::Matrix{T})
    s = size(m1)
    s[3] == size(m2,1) || throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(m2,1))"))
    hcat(map(k -> vcat(map(k1 -> (mapreduce(j -> readsegments(k1,j, m1)*slisemat(m2, s[1])[k][seg(j, s[1], s[3]),:], +, collect(1:s[2]))), collect(1:s[2]))...), collect(1:ceil(Int, size(m2,2)/s[1])))...)
end
 #bs times vector

function covbs{T <: AbstractFloat}(data::Matrix{T}, segments::Int = 2, corrected::Bool = false)
    ((size(data,2)%segments) <= (size(data,2)÷segments)) || throw(DimensionMismatch("last segment len $(size(data,2)-segments*(size(data,1)÷segments)) > segment len $(size(data,2)÷segments)"))
    BoxStructure(creatnarray(data, (b1::Int,b2::Int, data::Matrix{T}) -> cov(data[:,seg(b1, size(data,2)÷segments, size(data, 2))], data[:,seg(b2, size(data,2)÷segments, size(data, 2))], corrected = corrected), segments))
end

export BoxStructure, convert, matricise, trace, vec, *, square, vecnorm, +, covbs
end

# dokladnosci przy dodawaniu


