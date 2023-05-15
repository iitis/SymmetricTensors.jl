using Base.Cartesian
using ResumableFunctions

struct PyramidIndices{N}
    size::Int
end

@generated function _all_indices(p::PyramidIndices{N}) where {N}
    quote
        # multinds = Tuple{fill(Int, $N)...,}[]
        tensize = p.size
        @nloops $N i x -> (x==$N) ? (1:tensize) : (i_{x+1}:tensize) begin
            @inbounds multind = @ntuple $N x -> i_{$N-x+1}
            @yield multind
            # push!(multinds, multind)
        end
        # multinds
    end
end

function pi2(dims, tensize)
    p = PyramidIndices{dims}(tensize)
    return _all_indices(p)
end

a = rand(2,2,2)
dims, tensize = ndims(a), size(a, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)


b = rand(3,3,3)
dims, tensize = ndims(b), size(b, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)

@show pi2(dims, tensize)

c = rand(4,4,4,4)
dims, tensize = ndims(c), size(c, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)

@show "asdasd"
@time d = rand(10, 100, 100, 100, 100)
dims, tensize = ndims(d), size(d, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)

@time e = rand(20, 20, 20)
dims, tensize = ndims(e), size(e, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)


