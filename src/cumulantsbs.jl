# moments

# centre data
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

# calculate n'th moment in the bs form
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

#cumulants

# split indices into given permutation of partitions
function splitind(n::Vector{Int}, pe::Vector{Vector{Int}})
    ret = similar(pe)
    for k = 1:size(pe,1)
        @inbounds ret[k] = [map(i -> n[pe[k][i]], 1:size(pe[k],1))...]
    end
    ret
end

#if box is notsquared makes it square by adding slices with zeros
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
function productseg{T <: AbstractFloat}(s::Int, N::Int, part::Vector{Vector{Int}}, c::Array{T}...)
    ret = zeros(T, fill(s, N)...)
    for i = 1:(s^N)
        ind = ind2sub((fill(s, N)...), i)
        pe = splitind([ind...], part)
        @inbounds ret[ind...] = mapreduce(i -> c[i][pe[i]...], *, 1:size(part, 1))
    end
    ret
end


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

#checks if all bloks in bs are squred and call the proper function
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


#calculates all outer products of bs for given subsets of indices e.g. 6 -> 2,4
#provided all boxes in bs are squared
function pbcsquare{T <: AbstractFloat}(part::Vector{Int}, bscum::BoxStructure{T}...)
    N, s, n, p = partparameters(part, bscum...)
    ret = NullableArray(Array{T, N}, fill(s[2], N)...)
    ind = indices(N, s[2])
    for i in ind
      temp = zeros(T, fill(s[1], N)...)
      for pk in p
          pe = splitind([i...], pk)
          @inbounds temp += productseg(s[1], N, pk, map(i -> bscum[part[i]-1].frame[pe[i]...].value, 1:n)...)
      end
      @inbounds ret[i...] = temp
    end
    BoxStructure(ret)
end


#as above, but assumes last boxes in bs are not squared
function pbcnonsq{T <: AbstractFloat}(part::Vector{Int}, bscum::BoxStructure{T}...)
    N, s, n, p = partparameters(part, bscum...)
    ret = NullableArray(Array{T, N}, fill(s[2], N)...)
    ind = indices(N, s[2])
    block(i::Int, pe::Vector{Vector{Int}}) = bscum[part[i]-1].frame[pe[i]...].value
    for i in ind
      temp = zeros(T, fill(s[1], N)...)
      if (s[2] in i)
        block(i::Int, pe::Vector{Vector{Int}}) = addzeros(s[1], bscum[part[i]-1].frame[pe[i]...].value)
      end
      for pk in p
          pe = splitind([i...], pk)
          @inbounds temp += productseg(s[1], N, pk, map(i -> block(i, pe), 1:n)...)
      end
      if !(s[2] in i)
          @inbounds ret[i...] = temp
      else
          range = map(k -> ((s[2] == i[k])? (1:(s[3]%s[1])) : (1:s[1])), 1:size(i,1))
          @inbounds ret[i...] = temp[range...]
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

# calculates n'th cumulant
function cumulantn{T <: AbstractFloat}(m::Matrix{T}, n::Int, segments::Int, c::BoxStructure{T}...)
      ret = momentbc(m, n, segments)
      for p in findpart(n)
          ret -= pbc(p, c...)
      end
      ret
end

#recursive formula
function cumulants{T <: AbstractFloat}(n::Int, data::Matrix{T}, seg::Int = 2)
    data = centre(data)
    ret = Array(Any, n-1)
    for i = 2:n
      if i < 4
        ret[i-1] = momentbc(data, i, seg)
      else
        ret[i-1] = cumulantn(data, i, seg, ret[1:(i-3)]...)
      end
    end
    (ret...)
  end

