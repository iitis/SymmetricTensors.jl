# SymmetricTensors.jl
[![Build Status](https://travis-ci.org/ZKSI/SymmetricTensors.jl.svg?branch=master)](https://travis-ci.org/ZKSI/SymmetricTensors.jl)
[![Coverage Status](https://coveralls.io/repos/github/ZKSI/SymmetricTensors.jl/badge.svg?branch=master)](https://coveralls.io/github/ZKSI/SymmetricTensors.jl?branch=master)
[![DOI](https://zenodo.org/badge/79091776.svg)](https://zenodo.org/badge/latestdoi/79091776)

SymmetricTensors.jl provides the `SymmetricTensors{T, N}` type used to store fully symmetric tensors in more efficient way,
without most of redundant repetitions. It uses blocks of `Array{T, N}` stored in `NullableArrays{Array{T, N}, N}` type "https://github.com/JuliaStats/NullableArrays.jl".
Repeated blocks are replaced by #null. The module introduces `SymmetricTensors{T, N}` type and some basic operations on this type.
As of 01/01/2017 [@kdomino](https://github.com/kdomino) is the lead maintainer of this package.

## Installation

Within Julia, just use run

```julia
julia> Pkg.add("SymmetricTensors")
```

to install the files. Julia 0.6 or later is required.


## Constructor

```julia
julia> data = 2×2 NullableArrays.NullableArray{Array{Float64,2},2}:
[1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]
NULL               [1.0 1.0; 1.0 1.0]

julia> SymmetricTensor(data)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; NULL [1.0 1.0; 1.0 1.0]],2,2,4, true)
```

Without testing data symmetry and features

```julia
julia> SymmetricTensor(data; testdatstruct = false)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; NULL [1.0 1.0; 1.0 1.0]], 2, 2, 4, true)
```

## Converting

From `Array{T, N}` to `SymmetricTensors{T, N}`

```julia
julia> convert(SymmetricTensors, data::Array{T, N}, bls::Int = 2)
```
where bls is the size of a block


```julia
julia> data = ones(4,4)
4×4 Array{Float64,2}:
1.0  1.0  1.0  1.0
1.0  1.0  1.0  1.0
1.0  1.0  1.0  1.0
1.0  1.0  1.0  1.0

julia> convert(SymmetricTensor, data, 2)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; NULL [1.0 1.0; 1.0 1.0]], 2, 2, 4, true)
```

From `SymmetricTensors{T, N}` to `Array{T, N}`

```julia
julia> convert(Array, data::SymmetricTensors{T, N})
```


## Fields

- `frame::NullableArrays{Array{T, N}, N}`: stores data,
- `bls::Int`: size of a block,
- `bln::Int`: number of blocks,
- `dats::Int`: size of data,
- `sqr::Bool`: shows if the last block is squared.

## Operations

Addition and substraction: `+, -` is supported between two `SymmetricTensors{T, N}`. Addition substraction multiplication and division `+, -, *, /`
is supported between `SymmetricTensors{T, N}` and a number. For elementwise operation `f` between many `SymmetricTensors{T, N}` use `broadcast(f::Function, st::SymmetricTensors{T, N}...)`

The function diag returns a Vector{T}, of all super-diagonal elements of a SymmetricTensor.

```julia
julia> data = ones(5,5,5,5);

julia> st = convert(SymmetricTensor, data);

julia> diag(st)
5-element Array{Float64,1}:
 1.0
 1.0
 1.0
 1.0
 1.0
```

## Random Symmetric Ternsor generation

To generate random Symmetric Tensor just use `rand(SymmetricTensor{T, N}, dats::Int, bls::Int = 2)`.

```julia
julia> rand(SymmetricTensor{Float64, 2}, 2)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[0.587331 0.704768; 0.704768 0.836633]], 2, 1, 2, true)
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

This project was partially financed by the National Science Centre, Poland – project number 2014/15/B/ST6/05204.
