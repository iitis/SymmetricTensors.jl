Module SymmetricMatrix

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

function read_segments(data::BoxStructure ,i::Int, j::Int)
    a = data.frame
    test_index(a, [i,j])
    si = size(a[1,1],1)
    s = zeros(si, si)
    try
        s = a[i,j]
    catch
        s = a[j,i]'
    end
    return s
end

function into_segments{T <: AbstractFloat}(matrix::Array{T,2}, segment_numb::Int)
    issym(matrix)? true: error("imput matrix not symetric")
    s = size(matrix,1)
    (s%segment_numb == 0)? true: error("wrong number of segments")
    blockstruct = cell(segment_numb, segment_numb)
    ofset = Int(s/segment_numb)
    for i = 1:segment_numb, j = i:segment_numb
        blockstruct[i,j] = matrix[1+ofset*(i-1):ofset*i, 1+ofset*(j-1):ofset*j]
    end
    blockstruct
end


function multiplebs(m1::BoxStructure)
    blockstruct = cell(m1.sizeframe, m1.sizeframe)
    for k = 1:m1.sizeframe, l = k:m1.sizeframe
        res = zeros(m1.sizesegment, m1.sizesegment)
        for i = 1:m1.sizeframe
            res += read_segments(m1, k,i)*read_segments(m1, i,l)
        end
        blockstruct[k,l] = res
    end
    BoxStructure(blockstruct)
end

function multiplebs(m1::BoxStructure, m2::BoxStructure)
    (m1.sizeframe == m2.sizeframe)? true: error("different number of blocks")
    (m1.sizesegment == m2.sizesegment)? true: error("different size of blocks")
    msize = m1.sizeframe*m1.sizesegment
    ofset = m1.sizesegment
    matrix = zeros(msize, msize)
    for k = 1:m1.sizeframe, l = 1:m1.sizeframe
        res = zeros(m1.sizesegment, m2.sizesegment)
        for i = 1:m1.sizeframe
            res += read_segments(m1, k,i)*read_segments(m2, i,l)
        end
        matrix[((k-1)*ofset+1):(k*ofset),((l-1)*ofset+1):(l*ofset)] = res
    end
    matrix
end


function bstomatrix(m1::BoxStructure)
    msize = m1.sizeframe*m1.sizesegment
    matrix = zeros(msize,msize)
    ofset = m1.sizesegment
    for i = 1:m1.sizeframe, j = 1:m1.sizeframe
        matrix[((i-1)*ofset+1):(i*ofset),((j-1)*ofset+1):(j*ofset)] = read_segments(m1,i,j)
    end
    matrix
end

function gs(n::Int)
    A = randn(n,n)
    return A * A'
end
end
