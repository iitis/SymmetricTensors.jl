# ---- following code is used to caclulate moments ----

# centre data. Given data matrix centres each column,
#substracts columnwise mean for all data in the column
#performs centring foe each column, returns matrix
function centre{T<:AbstractFloat}(data::Matrix{T})
    centred = zeros(data)
    n = size(data, 2)
    for i = 1:n
        @inbounds centred[:,i] = data[:,i]-mean(data[:,i])
    end
    centred
end

# the single element of the block of N'th moment
momentel{T <: AbstractFloat}(v::Vector{T}...) = mean(mapreduce(i -> v[i], .*, 1:size(v,1)))

# calculate n'th moment for the given segment
function momentseg{T <: AbstractFloat}(r::Matrix{T}...)
    N = size(r, 1)
    dims = [map(i -> size(r[i], 2), 1:N)...]
    ret = zeros(T, dims...)
    for i = 1:mapreduce(k -> dims[k], *, 1:N)
        ind = ind2sub((dims...), i)
        @inbounds ret[ind...] = momentel(map(k -> r[k][:,ind[k]], 1:N)...)
    end
    ret
end

# calculate N'th moment in the bs form
#imput matrix of data, the order of the moemnt (N), number of degments for bs
#returns N dimentional Box structure of N'th moment
function momentbc{T <: AbstractFloat}(m::Matrix{T}, N::Int, segments::Int = 2)
    len = size(m,2)
    segsizetest(len, segments)
    (len%segments == 0)? () : segments += 1
    ret = NullableArray(Array{T, N}, fill(segments, N)...)
    segsize = ceil(Int, len/segments)
    ind = indices(N, segments)
    for i in ind
        @inbounds ret[i...] = momentseg(map(k -> m[:,seg(i[k], segsize, len)], 1:N)...)
    end
    BoxStructure(ret)
end

#--- following code is used to calculate cumulants ----

# split indices into given permutation of partitions
# imput: n , array of indices e.g. [i_1, i_2, i_3, i_4, i_5]
# permutation of partitions represented by followign integers e.g. [[2,3],[1,4,5]]
# output e.g. [[i_2 i_3][i_1 i_4 i_5]]
function splitind(n::Vector{Int}, pe::Vector{Vector{Int}})
    ret = similar(pe)
    for k = 1:size(pe,1)
        @inbounds ret[k] = [map(i -> n[pe[k][i]], 1:size(pe[k],1))...]
    end
    ret
end

#if box is notsquared makes it square by adding slices with zeros
#imput the box array and reguired size
#output N dimentional s x ...x s array
function addzeros{T <: AbstractFloat, N}(s::Int, imputbox::Array{T,N})
    if !all(collect(size(imputbox)) .== s)
        ret = zeros(T, fill(s, N)...)
        ind = map(k -> 1:size(imputbox,k), 1:N)
        ret[ind...] = imputbox
        return ret
    end
    imputbox
end

# calculates outer product of segments for given partition od indices
#input s - size of segment, N - required number of dinsions of output, part - vector of partations (vectors of ints)
# c - arrays of boxes
#returns N dimentional array of size s x .... x s
function productseg{T <: AbstractFloat}(s::Int, N::Int, part::Vector{Vector{Int}}, c::Array{T}...)
    ret = zeros(T, fill(s, N)...)
    for i = 1:(s^N)
        ind = ind2sub((fill(s, N)...), i)
        pe = splitind([ind...], part)
        @inbounds ret[ind...] = mapreduce(i -> c[i][pe[i]...], *, 1:size(part, 1))
    end
    ret
end

# sorts array of arrys of Ints according to the length of inner arrys
sortpart(ls::Vector{Vector{Int}}) = ls[sortperm(map(length, ls))]

# determines all permutations of [1,2,3, n] into given number of subsets
#and which element of the bs list correspond to such permutation
function partitionsind(part::Vector{Int})
    ret = Vector{Vector{Int}}[]
    for p in partitions(1:(cumsum(part)[end]), size(part,1))
        p = sortpart(p)
        if (mapreduce(i -> (size(p[i], 1) == part[i]), *, 1:size(part,1)))
            push!(ret, p)
        end
    end
    ret
end

#checks if all bloks in bs are squred and call the proper function that calculates mixed elements for the n'th cumulant
#imput part - a vector of partitions, bscum - cumulants of order 2 - (n-2) in bs form in the following order c2, c3, ..., c(n-2)
#output the porper outer product function
function pbc{T <: AbstractFloat}(part::Vector{Int}, bscum::BoxStructure{T}...)
  s = size(bscum[1])
  if (s[1]*s[2] == s[3])
    return pbcsquare(part, bscum...)
  else
    return pbcnonsq(part, bscum...)
  end
end

# calculates size type parameters given partition and sets of bs
partparameters{T <: AbstractFloat}(part::Vector{Int}, bscum::BoxStructure{T}...) = cumsum(part)[end], size(bscum[1]), size(part, 1), partitionsind(part)

# calculates sum of outer products at given multiindex and its partition
# condition boxes at given multiindex are not square
#imput p - particular partition, i - multiindex, bscumm - set od cumulants in bs structure
# output the sum of outer products at given multiindex i and given partition
function soopn{T <: AbstractFloat}(p::Vector{Vector{Vector{Int}}}, N::Int, n::Int, s::Tuple{Int64,Int64,Int64}, i::Vector{Int}, part::Vector{Int}, bscum::BoxStructure{T}...)
  temp = zeros(T, fill(s[1], N)...)
  for pk in p
      pe = splitind([i...], pk)
      @inbounds temp += productseg(s[1], N, pk, map(i -> addzeros(s[1], bscum[part[i]-1].frame[pe[i]...].value), 1:n)...)
  end
  range = map(k -> ((s[2] == i[k])? (1:(s[3]%s[1])) : (1:s[1])), 1:size(i,1))
  temp[range...]
end

# calculates sum of outer products at given multiindex and its partition
# condition boxes at given multiindex are square
#imput p - particular partition, i - multiindex, bscumm - set od cumulants in bs structure
# output the sum of outer products at given multiindex i and given partition
function soopsq{T <: AbstractFloat}(p::Vector{Vector{Vector{Int}}}, N::Int, n::Int, s::Tuple{Int64,Int64,Int64}, i::Vector{Int}, part::Vector{Int}, bscum::BoxStructure{T}...)
    temp = zeros(T, fill(s[1], N)...)
    for pk in p
        pe = splitind([i...], pk)
        @inbounds temp += productseg(s[1], N, pk, map(i -> bscum[part[i]-1].frame[pe[i]...].value, 1:n)...)
    end
    temp
end

#calculates the sum of all outer products at given partition of multiindex for all multiindexes for given bs
# condition all boxes in bs are squared
#imput part - a vector of partitions, bscum - cumulants of order 2 - (n-2) in bs form in the following order c2, c3, ..., c(n-2)
#output the sum of all outer products in the bs form
function pbcsquare{T <: AbstractFloat}(part::Vector{Int}, bscum::BoxStructure{T}...)
    N, s, n, p = partparameters(part, bscum...)
    ret = NullableArray(Array{T, N}, fill(s[2], N)...)
    ind = indices(N, s[2])
    for i in ind
      @inbounds ret[i...] = soopsq(p, N, n, s, i, part, bscum...)
    end
    BoxStructure(ret)
end


#calculates the sum of all outer products of bs for given partitions of indices if not all boxes in bs are squared
#imput part - a vector of partitions, bscum - cumulants of order 2 - (n-2) in bs form in the following order c2, c3, ..., c(n-2)
#output the sum of all outer products in the bs form
function pbcnonsq{T <: AbstractFloat}(part::Vector{Int}, bscum::BoxStructure{T}...)
    N, s, n, p = partparameters(part, bscum...)
    ret = NullableArray(Array{T, N}, fill(s[2], N)...)
    ind = indices(N, s[2])
    for i in ind
      if (s[2] in i)
        @inbounds ret[i...] = soopn(p, N, n, s, i, part, bscum...)
      else
        @inbounds ret[i...] = soopsq(p, N, n, s, i, part, bscum...)
      end
    end
    BoxStructure(ret)
end


# find all partitions of the order of cumulant into elements leq 2
function findpart(n::Int)
    ret = Array{Int, 1}[]
    for k = 2:floor(Int, n/2)
        for p in partitions(n,k)
            (1 in p)? (): push!(ret, sort(p))
        end
    end
    ret
end

# calculates n'th cumulant,
#imput data - matrix of data, n - the order of the cumulant, segments - number of segments for bs
# c - cumulants in the bs form orderred as follow c2, c3, ..., c(n-2)
#returns the n order cumulant in the bs form
function cumulantn{T <: AbstractFloat}(data::Matrix{T}, n::Int, segments::Int, c::BoxStructure{T}...)
      ret = momentbc(data, n, segments)
      for p in findpart(n)
          ret -= pbc(p, c...)
      end
      ret
end

#recursive formula, calculate cumulants up to order n
#imput: data - matrix of data, n - the maximal order of the cumulant, segments - number of segments for bs
#output: cumulants in the bs form orderred as follow c2, c3, ..., cn
function cumulants{T <: AbstractFloat}(n::Int, data::Matrix{T}, segments::Int = 2)
    data = centre(data)
    ret = Array(Any, n-1)
    for i = 2:n
      if i < 4
        ret[i-1] = momentbc(data, i, segments)
      else
        ret[i-1] = cumulantn(data, i, segments, ret[1:(i-3)]...)
      end
    end
    (ret...)
  end
