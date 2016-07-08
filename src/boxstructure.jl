
""" tests if the array is symmetric for a given tolerance

imput array, tolerance
"""
function issymetric{T <: AbstractFloat, N}(data::Array{T, N}, atol::Float64 = 1e-7)
  for i=2:ndims(data)
    (maximum(abs(unfold(data, 1)-unfold(data, i))) < atol) || throw(DimensionMismatch("array is not symmetric"))
  end
end

""" test wether the last segment of bs is not larger that an ordinary segment
"""
segsizetest(len::Int, segments::Int) = ((len%segments) <= (len÷segments)) || throw(DimensionMismatch("last segment len $len-segments*(len÷segments)) > segment len $(len÷segments)"))

"""examine if data can be stored in the bs form....

search for expected exception
"""
function structfeatures{T <: AbstractFloat, S}(frame::NullableArrays.NullableArray{Array{T,S},S})
  fsize = size(frame, 1)
  all(collect(size(frame)) .== fsize) || throw(DimensionMismatch("frame not square"))
  not_nulls = !frame.isnull
  !any(map(x->!issorted(ind2sub(not_nulls, x)), find(not_nulls))) || throw(ArgumentError("underdiagonal block not null"))
  quote
    @nloops $S i x->x==$S ? 1:fsize : i_{x+1}:fsize begin
      @inbounds minimum(size($frame[i].value)) .== size($frame[i].value, 1) || throw(DimensionMismatch("[$i ] block not square"))
    end
  end
  for i=1:fsize
    @inbounds issymetric(frame[fill(i, S)...].value)
  end
end

immutable BoxStructure{T <: AbstractFloat, S}
    frame::NullableArrays.NullableArray{Array{T,S},S}
    sizesegment::Int
    function call{T, S}(::Type{BoxStructure}, frame::NullableArrays.NullableArray{Array{T,S},S})
        structfeatures(frame)
        new{T, S}(frame, size(frame[fill(1,S)...].value,1))
    end
end

"""generates the set of sorted indices to run any operation on bs in a single loop.

imput N - number of dimentions, n - maximal index value

Return Array of indices (ints)
todo moze da sie zrobic w tuplach
"""
function indices(N::Int, n::Int)
    ret = Vector{Int}[]
    @eval begin
        @nloops $N i x -> (x==$N)? (1:$n): (i_{x+1}:$n) begin
            ind = @ntuple $N x -> i_{$N-x+1}
            @inbounds push!($ret, [ind...])
        end
    end
    ret
end

"""produces  set of indices for data in multidiemntional array
to read them in segments to perform bs

Return range
"""
seg(i::Int, of::Int, limit::Int) =  (i-1)*of+1 : ((i*of <= limit) ? i*of : limit)


"""converts super-symmetric array into bs

imput N dimentional Array

Returns N dimentional bs
"""
function convert{T <: AbstractFloat, N}(::Type{BoxStructure}, data::Array{T, N}, segments::Int = 2)
  issymetric(data)
  len = size(data,1)
  segsizetest(len, segments)
  (len%segments == 0)? () : segments += 1
  ret = NullableArray(Array{T, N}, fill(segments, N)...)
    ind = indices(N, segments)
    for writeind in ind
        readind = (map(k::Int -> seg(k, ceil(Int, len/segments), len), writeind)...)
        @inbounds ret[writeind...] = data[readind...]
    end
  BoxStructure(ret)
end

""" reads a segemnt with given multiindex,
if multiindex not sorted, find segment with sorted once and performs
  required permutation od fims

returns N dimentional Array
"""
function readsegments{T <: AbstractFloat, N}(i::Vector{Int}, bs::BoxStructure{T, N})
  sortidx = sortperm(i)
  permutedims(bs.frame[i[sortidx]...].value, invperm(sortidx))
end

"""gives the  number of boxes and the size of data stored in bs

Imput bs

Return Tuple of Ints
"""
function size{T <: AbstractFloat, N}(bsdata::BoxStructure{T, N})
  segsize = bsdata.sizesegment
  numsegments = size(bsdata.frame, 1)
  numdata = segsize * (numsegments-1) + size(bsdata.frame[end].value, 1)
  segsize, numsegments, numdata
end

"""tests if sizes on many bs are the same - for elementwise opertation on may bs
"""
function testsize{T <: AbstractFloat, N}(bsdata::BoxStructure{T, N}...)
  for i = 2:size(bsdata,1)
    @inbounds size(bsdata[1]) == size(bsdata[i]) || throw(DimensionMismatch("dims of B1 $(size(bsdata[1])) must equal to dims of B$i $(size(bsdata[i]))"))
  end
end

""" converts bs into Array
"""
function convert{T<:AbstractFloat, N}(::Type{Array}, bsdata::BoxStructure{T,N})
  s = size(bsdata)
  ret = zeros(T, fill(s[3], N)...)
    for i = 1:(s[2]^N)
        readind = ind2sub((fill(s[2], N)...), i)
        writeind = (map(k -> seg(readind[k], s[1], s[3]), 1:N)...)
        @inbounds ret[writeind...] = readsegments(collect(readind), bsdata)
      end
  ret
end

"""elementwise opertation on many bs

imput many bs of the same size

Returns single bs of the size of imput bs
"""
function operation{T<: AbstractFloat, N}(op::Function, bsdata::BoxStructure{T,N}...)
  n = size(bsdata, 1)
  (n > 1)? testsize(bsdata...):()
  ret = similar(bsdata[1].frame)
  ind = indices(N, size(bsdata[1].frame, 1))
  for i in ind
    @inbounds ret[i...] = op(map(k ->  bsdata[k].frame[i...].value, 1:n)...)
  end
  BoxStructure(ret)
end

"""elementwise opertation on bs and number

imput bs and number (Real)

Returns single bs of the size of imput bs
"""
function operation{T<: AbstractFloat, N}(op::Function, bsdata::BoxStructure{T,N}, a::Real)
  ret = similar(bsdata.frame)
  ind = indices(N, size(bsdata.frame, 1))
  for i in ind
    @inbounds ret[i...] = op(bsdata.frame[i...].value, a)
  end
  BoxStructure(ret)
end

"""elementwise opertation that changes the value of the bs (the f!() function )

imput bs and number (Real)

Returns single bs of the size of imput bs
"""
function operation!{T<: AbstractFloat,N, S <: Real}(bsdata::BoxStructure{T,N}, op::Function, n::S)
      ind = indices(N, size(bsdata.frame, 1))
      for i in ind
        @inbounds bsdata.frame[i...] = op(bsdata.frame[i...].value, n)
      end
end

# implements simple operations on bs structure


for op = (:+, :-, :.*, :./)
  @eval ($op){T <: AbstractFloat, N}(bsdata::BoxStructure{T, N}, bsdata1::BoxStructure{T, N}) = operation($op, bsdata, bsdata1)
end


for op = (:+, :-, :*, :/)
  @eval ($op){T <: AbstractFloat, S <: Real}(bsdata::BoxStructure{T}, n::S)  = operation($op, bsdata, n)
end

"""add function that changes the imput data f!() type

imput bs data to which a number is added elementwisely
"""
add{T <: AbstractFloat, S <: Real}(bsdata::BoxStructure{T}, n::S)  = operation!(bsdata, +, n)
