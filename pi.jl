using Base.Cartesian

function pi2(dims::Int, tensize::Int)
    multinds = Tuple{fill(Int, dims)...,}[]
    quote
        @nloops $dims i x -> (x==dims) ? (1:tensize) : (i_{x+1}:tensize) begin
            @inbounds multind = @ntuple $dims x -> i_{dims-x+1}
            push!($multinds, multind)
        end
    end
    multinds
end


a = rand(2,2,2)
dims, tensize = ndims(a), size(a, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)


b = rand(3,3,3)
dims, tensize = ndims(b), size(b, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)

c = rand(4,4,4,4)
dims, tensize = ndims(c), size(c, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)

@time d = rand(10, 100, 100, 100, 100)
dims, tensize = ndims(d), size(d, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)

@time e = rand(20, 20, 20)
dims, tensize = ndims(e), size(e, 1)
@time pi2(dims, tensize)
@time pi2(dims, tensize)


@show pi2(dims, tensize)