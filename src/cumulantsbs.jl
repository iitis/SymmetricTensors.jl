# ---- following code is used to caclulate moments ----

""" centre data. Given data matrix centres each column,
substracts columnwise mean for all data in the column
performs centring for each column,

Returns matrix
"""
function centre!{T<:AbstractFloat}(data::Matrix{T})
  n = size(data, 2)
  for i = 1:n
    @inbounds data[:,i] = data[:,i]-mean(data[:,i])
  end
end

function centre{T<:AbstractFloat}(data::Matrix{T})
  centred = copy(data)
  centre!(centred)
  centred
end

""" calculates the single element of the block of N'th moment

input vectors that corresponds to given column of data

Returns Float64 (an element of the block)
"""
momentel{T <: AbstractFloat}(v::Vector{T}...) = mean(mapreduce(i -> v[i], .*, 1:length(v)))

"""calculate n'th moment for the given segment

input r - matrices of data

Returns N dimentional array (segment)
"""
function momentseg{T <: AbstractFloat}(dims::Array{Int}, Y::Matrix{T}...)
  n = length(Y)
#  ret = SharedArray(T, dims...)
  ret = zeros(T, dims...)
#  @sync @parallel
  for i = 1:prod(dims)
    ind = ind2sub((dims...), i)
    @inbounds ret[ind...] = momentel(map(k -> Y[k][:,ind[k]], 1:n)...)
  end
  ret
end

"""axilaiary function that creates indices for blocks
"""
sqseg(i::Int, of::Int) =  (i-1)*of+1 : i*of


""" calculate N'th moment in the bs form

input matrix of data, the order of the moment (N), number of segments for bs

Returns N dimentional Box structure of N'th moment

Particular case if all boxes are squared
"""
function centrmom{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int)
    g = div(size(X,2), s)
    ret = NullableArray(Array{T, n}, fill(g, n)...)
    dims = fill(s, n)
    for i in indices(n, g)
      @inbounds ret[i...] = momentseg(dims, map(k -> X[:,sqseg(i[k], s)], 1:n)...)
    end
    BoxStructure(ret)
end


""" calculate N'th moment in the bs form

input matrix of data, the order of the moment (N), number of segments for bs

Returns N dimentional Box structure of N'th moment

Case if last boxes are not squared
"""
function centrmomnsq{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int)
    M = size(X,2)
    g = ceil(Int, M/s)
    ret = NullableArray(Array{T, n}, fill(g, n)...)
    for i in indices(n, g)
      Y = map(k -> X[:,seg(i[k], s, M)], 1:n)
      dims = map(i -> (size(Y[i], 2)), 1:n)
      @inbounds ret[i...] = momentseg(dims, Y...)
    end
    BoxStructure(ret)
end

"""examines if the last box is square or not

and call the proper function
"""
function momentbs{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int = 2)
    (size(X,2)%s == 0)? centrmom(X,n,s) : centrmomnsq(X,n,s)
end

#--- following code is used to calculate cumulants ----

"""split indices into given permutation of partitions

input: n , array of indices e.g. [i_1, i_2, i_3, i_4, i_5]
permutation of partitions represented by followign integers e.g. [[2,3],[1,4,5]]

Returns output e.g. [[i_2 i_3][i_1 i_4 i_5]]
"""
splitind(n::Vector{Int}, pe::Vector{Vector{Int}}) = map(p->n[p], pe)


"""calculates outer product of segments for given partition od indices

input s - size of segment, N - required number of dinsions of output, part - vector of partations (vectors of ints)
c - arrays of boxes

Returns N dimentional array of size s x .... x s
"""
function prodblocks{T <: AbstractFloat}(s::Int, n::Int, part::Vector{Vector{Int}}, c::Array{T}...)
  ret = zeros(T, fill(s, n)...)
  for i = 1:(s^n)
    ind = ind2sub((fill(s, n)...), i)
    pe = splitind(collect(ind), part)
    @inbounds ret[ind...] = mapreduce(i -> c[i][pe[i]...], *, 1:size(part, 1))
    #size(c, 1) = size(part, 1)
  end
  ret
end


""" given to multiindex lenght n and number of subsets omega
provides all partitions of the sequence 1:n into sigma subsets such that each
subset size is greater than 1 and subsets are disjoint
"""

function indpart(n::Int, sigma::Int)
    p = Vector{Vector{Int}}[]
    r = Vector{Int}[]
    for part in partitions(1:n, sigma)
      s = map(length, part)
      if !(1 in s)
        push!(p, part)
        push!(r, s)
      end
    end
    p, r, length(r)
end


"""if box is notsquared makes it square by adding slices with zeros

input the box array and reguired size

Returns N dimentional s x ...x s array
"""
function addzeros{T <: AbstractFloat, N}(s::Int, inputbox::Array{T,N})
  if !all(collect(size(inputbox)) .== s)
    ret = zeros(T, fill(s, N)...)
    ind = map(k -> 1:size(inputbox,k), 1:N)
    ret[ind...] = inputbox
    return ret
  end
  inputbox
end

"""
calculates mixed element for given sigma, if all blockes are squared

input n - order of product, sigma - number of subsets,
c2, c3, ..., c(n-2) - lowe cumulants input

Returns the porper outer product function
"""
function outerprod{T <: AbstractFloat}(n::Int, sigma::Int, c::BoxStructure{T}...)
  s,g,M = size(c[1])
  p, r, len = indpart(n, sigma)
  ret = NullableArray(Array{T, n}, fill(g, n)...)
  for i in indices(n, g)
    temp = zeros(T, fill(s, n)...)
    for j in 1:len
      pe = splitind([i...], p[j])
      @inbounds temp += prodblocks(s, n, p[j], map(l -> c[r[j][l]-1].frame[pe[l]...].value, 1:sigma)...)
    end
    @inbounds ret[i...] = temp
  end
  BoxStructure(ret)
end

"""
calculates mixed element for given sigma, if last blockes are not squared

"""
function outerprodnsq{T <: AbstractFloat}(n::Int, sigma::Int, c::BoxStructure{T}...)
  s,g,M = size(c[1])
  p, r, len = indpart(n, sigma)
  ret = NullableArray(Array{T, n}, fill(g, n)...)
  for i in indices(n, g)
    temp = zeros(T, fill(s, n)...)
    for j in 1:len
      pe = splitind([i...], p[j])
      if (g in i)
        @inbounds temp += prodblocks(s, n, p[j], map(l -> addzeros(s[1], c[r[j][l]-1].frame[pe[l]...].value), 1:sigma)...)
      else
        @inbounds temp += prodblocks(s, n, p[j], map(l -> c[r[j][l]-1].frame[pe[l]...].value, 1:sigma)...)
      end
    end
    if (g in i)
      range = map(k -> ((g == i[k])? (1:(M%s)) : (1:s)), 1:length(i))
      temp = temp[range...]
    end
    @inbounds ret[i...] = temp
  end
  BoxStructure(ret)
end


"""checks if all bloks in are squred and returns

the proper function that calculates mixed elements for the n'th cumulant
"""
function outerp{T <: AbstractFloat}(n::Int, sigma::Int, c::BoxStructure{T}...)
  s,g,M = size(c[1])
  (M%s == 0)? outerprod(n,sigma,c...) : outerprodnsq(n,sigma,c...)
end

"""calculates n'th cumulant,

input data - matrix of data, n - the order of the cumulant, segments - number of segments for bs
c - cumulants in the bs form orderred as follow c2, c3, ..., c(n-2)

Returns the n order cumulant in the bs form"""
function cumulantn{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int, c::BoxStructure{T}...)
  ret =  momentbs(X, n, s)
  for sigma in 2:floor(Int, n/2)
    ret -= outerp(n, sigma, c...)
  end
  ret
end


"""recursive formula, calculate cumulants up to order n

input: data - matrix of data, n - the maximal order of the cumulant, segments - number of segments for bs

Returns cumulants in the bs form orderred as follow c2, c3, ..., cn
works for any n >= 2, tested up to n = 10, in automatic tests up to n = 6 (limit due to the increasement
in computation time for benchmark algorithm (semi naive))
"""
function cumulants{T <: AbstractFloat}(n::Int, X::Matrix{T}, s::Int = 3)
  X = centre(X)
  ret = Array(BoxStructure{T}, n-1)
  for i = 2:n
    ret[i-1] =  (i < 4)? momentbs(X, i, s) : cumulantn(X, i, s, ret[1:(i-3)]...)
  end
  ret
end
