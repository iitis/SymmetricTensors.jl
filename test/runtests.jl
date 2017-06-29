using FactCheck
using SymmetricTensors
using NullableArrays
using Combinatorics
using Iterators

import SymmetricTensors: ind2range, indices, accesnotord, issymetric, sizetest


facts("Helpers") do
  A = reshape(collect(1.:8.), 2, 2, 2)
  context("unfold") do
    @fact unfold(A, 1) --> [[1. 3. 5. 7.]; [2. 4. 6. 8.]]
    @fact unfold(A, 2) --> [[1. 2. 5. 6.]; [3. 4. 7. 8.]]
    @fact unfold(A, 3) --> [[1. 2. 3. 4.]; [5. 6. 7. 8.]]
  end
  context("issymmetric") do
    A = reshape(collect(1.:8.), 2, 2, 2)
    @fact_throws AssertionError issymetric(A)
    @fact issymetric([[1. 2.]; [2. 1.]]) --> nothing
  end
  context("indexing") do
    @fact indices(2,3) --> [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]
    @fact ind2range(2,3,5) --> 4:5
  end
  context("sizetest") do
    @fact_throws DimensionMismatch sizetest(2,3)
  end
end

# generates symmetric tensors
srand(42)
t = zeros(7,7,7)
t1 = zeros(7,7,7)
for i in indices(3,7)
  x1 = rand()
  x2 = rand()
  for j in collect(permutations(i))
    t[j...] = x1
    t1[j...] = x2
  end
end

facts("Converting") do
  b = convert(SymmetricTensor, t[1:6, 1:6, 1:6], 3)
  context("Acessing Symmetric Tensors") do
    @fact b[(1,1,1)] --> t[1:3, 1:3, 1:3]
    @fact accesnotord(b, (2,1,2)) --> t[4:6, 1:3, 4:6]
    @fact accesnotord(b, (2,1,1)) --> t[4:6, 1:3, 1:3]
  end
  context("converting from array to SymmetricTensor") do
    a = reshape(collect(1.:16.), 4, 4)
    @fact convert(SymmetricTensor, a*a')[1,1] -->  [276.0  304.0; 304.0  336.0]
    @fact b.frame[1,1,1].value --> roughly(t[1:3, 1:3, 1:3])
    @fact b.frame[1,2,2].value --> roughly(t[1:3, 4:6, 4:6])
    @fact b.frame[2,2,2].value --> roughly(t[4:6, 4:6, 4:6])
    @fact isnull(b.frame[2,1,1]) --> true
  end
  context("Constructor tests") do
    b1 = convert(SymmetricTensor, t[1:6, 1:6, 1:6], 4)
    @fact b.sqr --> true
    @fact b1.sqr --> false
    @fact b.bls --> 3
    @fact b.bln --> 2
    @fact b.dats --> 6
  end
end

facts("Basic operations") do
  b = convert(SymmetricTensor, t)
  b1 = convert(SymmetricTensor, t1)
  context("Get super-diagonal") do
    @fact diag(b) --> roughly([t[fill(i,ndims(t))...] for i in 1:size(t,1)])
    @fact diag(b1) --> roughly([t1[fill(i,ndims(t1))...] for i in 1:size(t1,1)])
  end
  context("Elementwise operations") do
    @fact convert(Array,b+b1) --> roughly(t+t1)
    @fact convert(Array,b-b1) --> roughly(t-t1)
    @fact convert(Array,b.*b1) --> roughly(t.*t1)
    @fact convert(Array,b./b1) --> roughly(t./t1)
  end
  context("Matrix--scalar operations") do
    @fact convert(Array,b*2.1) -->roughly(t*2.1)
    @fact convert(Array,b/2.1) -->roughly(t/2.1)
    @fact convert(Array,b/2) -->roughly(t/2)
    @fact convert(Array,b+2.1) -->roughly(t+2.1)
    @fact convert(Array,b-2.1) -->roughly(t-2.1)
    @fact convert(Array,b+2) -->roughly(t+2)
    @fact convert(Array,2+b) -->roughly(t+2)
  end
end

facts("Exceptions") do
  context("Dimensions in operations") do
    b = convert(SymmetricTensor, t)
    b1 = convert(SymmetricTensor, t, 3)
    b2 = convert(SymmetricTensor, t[1:6, 1:6, 1:6])
    b2 = convert(SymmetricTensor, t[:,:,1])
    @fact_throws DimensionMismatch b+b1
    @fact_throws MethodError b+b2
    @fact_throws UndefVarError b+b3
  end

  context("Constructor exceptions") do
    @fact_throws TypeError SymmetricTensor([1. 2.];[3. 4.])
    @fact_throws DimensionMismatch SymmetricTensor(t[:, :, 1:2])
    b = SymmetricTensor(t).frame
    b1 = copy(b)
    b2 = copy(b)
    b[1,1,1] = reshape(collect(1:8), (2,2,2))
    @fact_throws AssertionError SymmetricTensor(b)
    b1[1,2,3] = reshape(collect(1:4), (2,2,1))
    @fact_throws AssertionError SymmetricTensor(b1)
    b2[3,2,1] = reshape(collect(1:8), (2,2,2))
    @fact_throws AssertionError SymmetricTensor(b2)
    # wrong block size
    @fact_throws DimensionMismatch convert(SymmetricTensor, t,  25)
    @fact_throws DimensionMismatch convert(SymmetricTensor, t,  0)
  end
end
