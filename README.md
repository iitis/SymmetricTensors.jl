# SymmetricTensors.jl
[![Coverage Status](https://coveralls.io/repos/github/iitis/SymmetricTensors.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/SymmetricTensors.jl?branch=master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7034097.svg)](https://doi.org/10.5281/zenodo.7034097)

SymmetricTensors.jl provides the `SymmetricTensors{T, N}` type used to store fully symmetric tensors in more efficient way,
without most of redundant repetitions. It uses blocks of `Array{T, N}` stored in `Union{Array{Float,N}, Nothing}` structure.
Repeated blocks are replaced by `Void`. The module introduces `SymmetricTensors{T, N}` type and some basic operations on this type.
As of 01/01/2017 [@kdomino](https://github.com/kdomino) is the lead maintainer of this package.

## Installation

Within Julia, just use run

```julia
pkg> add SymmetricTensors
```

to install the files. Julia 1.0 or later is required.


## Constructor

```julia
julia> data = ones(4,4);


julia> SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; nothing [1.0 1.0; 1.0 1.0]], 2, 2, 4, true)


```

## Converting

From `Array{T, N}` to `SymmetricTensors{T, N}`

```julia
julia> SymmetricTensors(data::Array{T, N}, bls::Int = 2)
```
where bls is the size of a block. It is a parameter affecting the compuational speed of cumulants. The block size must fulfill `bls ∈ {1, 2,..., dats}` where `dats = size(data, 1) = ... = size(data, N)` otherwise error is risen.


```julia
julia> data = ones(4,4);


julia> convert(SymmetricTensor, data, 2)
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; nothing [1.0 1.0; 1.0 1.0]], 2, 2, 4, true)

```

From `SymmetricTensors{T, N}` to `Array{T, N}`

```julia
julia> Array(st::SymmetricTensors{T, N})
```

Wrong block size:

```
julia> SymmetricTensor(ones(4,4), 5)
ERROR: DimensionMismatch("bad block size 5 > 4")
```


## Fields

- `frame::ArrayNArrays{T,N}`: stores data, where `ArrayNArrays{T,N} = Array{Union{Array{T, N}, Nothing}, N}`
- `bls::Int`: the size of ordinary block (the same in each direction),
- `bln::Int`: maximal number of blocks in each direction,
- `dats::Int`: the size of data stored (the same in each direction),
- `sqr::Bool`: if all blocks are squares (N-squares).


Suppose we have `N = 2` and `dats = 6` and `bls = 3` hence data are symmetric matrix of size `6 x 6`. Data are stored in the form:

```
|B11   B12 | 
|null  B22 | 
```

here `bls = 2` and size of `B11`, `B12`, and `B22` are `3 x 3`. Bear in mind, that `B11` and `B22` his to be symmetric. As `B12` (the last block) is square, `sqr = True`.

Suppose now `N = 2` and `dats = 5` and `bls = 3` hence data are symmetric matrix of size `5 x 5`. Data are stored in similar form:

|B11   B12 | 

|null  B22 | 

here `bls = 2` and size of `B11` is `3 x 3`, but size of `B12` is `2 x 3`, and size `B22` is `2 x 2 `. Again `B11` and `B22` his to be symmetric. As `B12` (the last block) is not square, `sqr = False`.

For `N = 3` we have analogical pyramid representation, and for `N > 3` hyper-pyramid representation.



## Operations

Elementwise addition: `+, -` is supported between many `SymmetricTensors{T, N}` while elementwise substraction: `-` between two `SymmetricTensors{T, N}`. Addition substraction multiplication and division `+, -, *, /`
is supported between `SymmetricTensors{T, N}` and a number. 

```julia
julia> x = SymmetricTensor(ones(4,4));

julia> y = SymmetricTensor(2*ones(4,4));

julia> x+y
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[3.0 3.0; 3.0 3.0] [3.0 3.0; 3.0 3.0]; #undef [3.0 3.0; 3.0 3.0]], 2, 2, 4, true)

julia> x*10
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[10.0 10.0; 10.0 10.0] [10.0 10.0; 10.0 10.0]; #undef [10.0 10.0; 10.0 10.0]], 2, 2, 4, true)
```


The function diag returns a Vector{T}, of all super-diagonal elements of a SymmetricTensor.

```julia
julia> data = ones(5,5,5,5);

julia> st = SymmetricTensor(data);

julia> diag(st)
5-element Array{Float64,1}:
 1.0
 1.0
 1.0
 1.0
 1.0
```

## Random Symmetric Tensor generation

To generate random Symmetric Tensor with random elements of typer `T` form a uniform distribution on `[0,1)` use `rand(SymmetricTensor{T, N}, n::Int, b::Int = 2)`. Here `n` denotes size of each mode and `b` denotes block size. Eg. for `N = 4` we would have `n x n x n x n` tensor.

```julia
julia> using Random

julia> Random.seed!(42)

julia> x = rand(SymmetricTensor{Float64, 3}, 2)
SymmetricTensor{Float64, 3}(Union{Nothing, Array{Float64, 3}}[[0.5331830160438613 0.4540291355871424; 0.4540291355871424 0.017686826714964354]

[0.4540291355871424 0.017686826714964354; 0.017686826714964354 0.17293302893695128]], 2, 1, 2, true)

julia> Array(x)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.533183  0.454029
 0.454029  0.0176868

[:, :, 2] =
 0.454029   0.0176868
 0.0176868  0.172933


```

```julia
julia> Random.seed!(42)

julia> x = rand(SymmetricTensor{Float64, 2}, 3)
SymmetricTensor{Float64, 2}(Union{Nothing, Matrix{Float64}}[[0.5331830160438613 0.4540291355871424; 0.4540291355871424 0.017686826714964354] [0.17293302893695128; 0.9589258763297348]; nothing [0.9735659798036858]], 2, 2, 3, false)

julia> Array(x)
3×3 Matrix{Float64}:
 0.533183  0.454029   0.172933
 0.454029  0.0176868  0.958926
 0.172933  0.958926   0.973566

```

## getindex and setindex!

```julia
julia> using Random

julia> Random.seed!(42)

julia> st = rand(SymmetricTensor{Float64, 2}, 2)
SymmetricTensor{Float64,2}(Union{Nothing, Array{Float64,2}}[[0.533183 0.454029; 0.454029 0.0176868]], 2, 1, 2, true)

julia> st[1,2]
0.4540291355871424

julia> st[2,1]
0.4540291355871424


julia> pyramidindices(st)
3-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (1, 2)
 (2, 2)

```
Function ```pyramidindices(st::SymmetricTensor)``` returns the indices of the unique element of the give symmetric tensor


`setindex!(st::SymmetricTensor, x::Float, mulind::Int...)` changes all symmetric tensor's elements indexed by `mulind` to `x`.

```julia
julia> st[1,2] = 10.

julia> convert(Array, st)
2×2 Array{Float64,2}:
  0.533183  10.0      
 10.0        0.0176868

```

## Auxiliary function

```julia
julia> unfold(data::Array{T,N}, mode::Int)
```
unfolds `data` in a given mode

```julia
julia> a = reshape(collect(1.:8.), (2,2,2))
2×2×2 Array{Float64,3}:
[:, :, 1] =
 1.0  3.0
 2.0  4.0

[:, :, 2] =
 5.0  7.0
 6.0  8.0

julia> unfold(a, 1)
2×4 Array{Float64,2}:
 1.0  3.0  5.0  7.0
 2.0  4.0  6.0  8.0

julia> unfold(a, 2)
2×4 Array{Float64,2}:
 1.0  2.0  5.0  6.0
 3.0  4.0  7.0  8.0

julia> unfold(a, 3)
2×4 Array{Float64,2}:
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
```

## Block structure

The block usage is motivated by the paper M. D. Schatz, T. M. Low, R. A. van de Geijn, and T. G. Kolda, "Exploiting symmetry in tensors for high performance: Multiplication with symmetric tensors", SIAM Journal on Scientific Computing, 36 (2014), pp. C453–C479 https://doi.org/10.1137/130907215. There only the meaningful part of the symmetric tensor is stored in blocks to decrease the memory and computational overhead. The selection of the optimal block size is not straight forward, however in most cases concerning cumulants one can use `2` or `3`.



This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.
