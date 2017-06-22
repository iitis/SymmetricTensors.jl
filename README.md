# SymmetricTensors.jl
[![Build Status](https://travis-ci.org/kdomino/SymmetricTensors.jl.svg?branch=master)](https://travis-ci.org/kdomino/SymmetricTensors.jl)
[![Coverage Status](https://coveralls.io/repos/github/ZKSI/SymmetricTensors.jl/badge.svg?branch=master)](https://coveralls.io/github/ZKSI/SymmetricTensors.jl?branch=master)

SymmetricTensors.jl provides the `SymmetricTensors{T, N}` type used to store fully symmetric tensors in more efficient way,
without most of redundant repetitions. Uses blocks of `Array{T, N}` stored in `NullableArrays{Array{T, N}, N}` type "https://github.com/JuliaStats/NullableArrays.jl".
Repeating blocks are replaced by #null. The module introduces `SymmetricTensors{T, N}` type and some basic operations on this type.
As of 01/01/2017 "https://github.com/kdomino" is the lead maintainer of this package.

## Installation

Within Julia, just use run

```julia
julia> Pkg.clone("https://github.com/kdomino/SymmetricTensors.jl")
```

to install the files.  Julia 0.5 or later required.


## Constructor

```julia
julia> data = 2×2 NullableArrays.NullableArray{Array{Float64,2},2}:
[1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]
#NULL               [1.0 1.0; 1.0 1.0]

julia> SymmetricTensor(data)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; #NULL [1.0 1.0; 1.0 1.0]],2,2,4,true)       
```

Without testing data symmetry and features

```julia
julia> julia> SymmetricTensor(data; testdatstruct = false)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; #NULL [1.0 1.0; 1.0 1.0]],2,2,4,true)
```

## Converting

From `Array{T, N}` to `SymmetricTensors{T, N}`

```julia
julia> convert(SymmetricTensors, data::Array{T, N}, bls::Int = 2)
```
where bls is a size of a block


```julia
julia> data = ones(4,4)
4×4 Array{Float64,2}:
1.0  1.0  1.0  1.0
1.0  1.0  1.0  1.0
1.0  1.0  1.0  1.0
1.0  1.0  1.0  1.0

julia> convert(SymmetricTensor, data, 2)
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[1.0 1.0; 1.0 1.0] [1.0 1.0; 1.0 1.0]; #NULL [1.0 1.0; 1.0 1.0]],2,2,4,true)
```

From `SymmetricTensors{T, N}` to `Array{T, N}`

```julia
julia> convert(Array, data::SymmetricTensors{T, N})
```


## Fields

- `frame::NullableArrays{Array{T, N}, N}`: stores data,
- `bls::Int`: size of a block,
- `bln::Int`: number of blocks,
- `datas::Int`: size of data,
- `sqr::Bool`: is last block squared.

## Operations

Following element-wise operations `+, -, *, .*, /, ./` are supporter between two `SymmetricTensors{T, N}` objects or a `SymmetricTensors{T, N}` object and a number.

## Axiliary function

```julia
julia> unfold(data::Array{T,N}, mode::Int)
```
unfolds array in a given mode

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
