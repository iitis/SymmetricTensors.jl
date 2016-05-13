module SymmetricMatrix
using NullableArrays
import Base: trace, vec, vecnorm, +, *, .*, size

seg(i::Int, ofset::Int) = ((i-1)*ofset+1):(i*ofset)

function creatnarray{T <: AbstractFloat}(data::Matrix{T}, rule::Function, segments::Int)
    ret = NullableArray(Matrix{T}, segments, segments)
    for i = 1:segments, j = i:segments
        ret[i,j] = rule(i,j, data)::Matrix{T}
    end
    ret
end

function segmentise{T <: AbstractFloat}(data::Matrix{T}, segments::Int)
    (maximum(abs(data - transpose(data))) < 1e-7) || throw(DimensionMismatch("matrix is not symmetric")) #poprawic
    (size(data,1)%segments == 0) || throw(DimensionMismatch("data size $(size(data,1)) / segment size $segments not integer"))
    creatnarray(data, (i::Int,j::Int, data::Matrix{T}) -> data[seg(i, size(data,1)÷segments), seg(j, size(data,1)÷segments)], segments)
end


function structfeatures{T <: AbstractFloat}(frame::NullableArrays.NullableArray{Matrix{T},2})
    isequal(size(frame)...) || throw(DimensionMismatch("frame not symmetric"))
    for i = 1:size(frame, 1), j = 1:size(frame, 1)
        (i > j)? isnull(frame[i,j]) || throw(ArgumentError("underdiagonal block [ $i , $j ] not null")): isequal(size(frame[i,j].value)...) || throw(DimensionMismatch("[ $i , $j ] block not squared"))       
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

slisemat{T <: AbstractFloat}(m2::Matrix{T}, slisesize::Int) = map(i -> m2[:,seg(i, slisesize)], collect(1:size(m2,2)÷slisesize))

function *{T <: AbstractFloat}(m1::BoxStructure{T}, m2::Matrix{T})
    s = size(m1)
    s[3] == size(m2,1) || throw(DimensionMismatch("size of B1 $(s[3]) must equal to size of A $(size(m2,1))"))
    size(m2,2)%s[1] == 0 || throw(DimensionMismatch(" matrix size $(size(m2,2)) / segment size $(s[1]) not integer"))
    hcat(map(k -> vcat(map(k1 -> (mapreduce(j -> readsegments(k1,j, m1)*slisemat(m2, s[1])[k][seg(j, s[1]),:], +, collect(1:s[2]))), collect(1:s[2]))...), collect(1:size(m2,2)÷s[1]))...)
end

function covbs{T <: AbstractFloat}(datatab::Matrix{T}, blocks::Int = 2, corrected::Bool = false)
    size(datatab, 2)%blocks == 0 || throw(DimensionMismatch("data size $(size(datatab, 2)) / segment size $blocks not integer"))
    BoxStructure(creatnarray(datatab, (b1::Int,b2::Int, data::Matrix{T}) -> cov(data[:,seg(b1, blocks)], data[:,seg(b2, blocks)], corrected = corrected), size(datatab, 2)÷blocks))
end

export BoxStructure, convert, matricise, trace, vec, *, square, vecnorm, +, covbs
end

# dokladnosci przy dodawaniu


