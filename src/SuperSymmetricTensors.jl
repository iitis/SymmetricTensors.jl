module SuperSymmetricTensors

using Tensors
immutable BCSS{T<:Real}
    data::Dict{Tuple{Int, Int}, Matrix{T}}
    rows::Int
    columns::Int
    blocksize::Int
end

function BCSS{T<:Real}(mat::Symmetric{T}, blocksize::Int)
    rows = size(mat, 1)÷blocksize
    columns = size(mat, 2)÷blocksize
    data = Dict()
    for i=1:rows, j=1:cols
        data[(i,j)]=mat[i:blocksize:i+1, i:blocksize:i+1]
    end
    BCSS(data)
end
function getblockelement{T}(mat::BCSS{T}, i::Int, j::Int)
    return mat.data[(i,j)] 
end
end
