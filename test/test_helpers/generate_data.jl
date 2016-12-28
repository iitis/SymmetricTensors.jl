# generate data for tests
srand(42)
"""

  symmetrise(m::Array)

Returns array that is symmetric in all modes

```jldoctest
julia> A = [1. 2. ; 3. 4.];

julia> symmetrise(A)
2×2 Array{Float64,2}:
 1.0  2.0
 2.0  4.0
```
"""
function symmetrise(m::Array)
  symarray = zeros(m)
  for multind in product(fill(collect(1:size(m, 1)), ndims(m))...)
    if issorted(multind)
      for k in collect(permutations(multind))
        symarray[k...] = m[multind...]
      end
    end
  end
  symarray
end

"""

  generatedata(dats::Int = 15, bls::Int = 4, dims::Int = 3)

Returns array{dims},  symmetric array{dims}, symmetric array{dims} in
Symmetric Tensors form
"""
function generatedata(dats::Int = 15, bls::Int = 4, dims::Int = 3)
    rmat = randn(fill(dats, dims)...)
    rmat, symmetrise(rmat), convert(SymmetricTensor, symmetrise(rmat), bls)
 end


"""
Returns NullableArray of arrays generating exceptions on type constructor

nonull_el = false, nonsq_box = false - nullable array with non symmetric diagonal
nonull_el = true - no nulls below diagonal at index [3,2,1]
nonull_el = false, nonsq_box = true - block at index [1,2,3] not squaerd
"""
 function create_except(dat::Array, nonull_el::Bool = false, nonsq_box::Bool = false)
  dims = ndims(dat)
  nullablearr = NullableArray(Array{Float64, dims}, fill(4, dims)...)
  for i in product(fill(1:4, dims)...)
    issorted(i)? nullablearr[i...] = dat : ()
  end
  if nonull_el
    nullablearr[reverse(collect(1:dims))...] = dat
  elseif nonsq_box
    nullablearr[collect(1:dims)...] = dat[1:2,:]
  end
  nullablearr
end
