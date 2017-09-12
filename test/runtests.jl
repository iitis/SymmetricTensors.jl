using Base.Test

using SymmetricTensors
using NullableArrays

import SymmetricTensors: ind2range, indices, issymetric, sizetest,
getblock, getblockunsafe, broadcast, randsymarray


@testset "Helpers" begin
  A = reshape(collect(1.:8.), 2, 2, 2)
  @testset "unfold" begin
    @test unfold(A, 1) == [[1.0 3.0 5.0 7.0]; [2.0 4.0 6.0 8.0]]
    @test unfold(A, 2) == [[1.0 2.0 5.0 6.0]; [3.0 4.0 7.0 8.0]]
    @test unfold(A, 3) == [[1.0 2.0 3.0 4.0]; [5.0 6.0 7.0 8.0]]
  end
  @testset "issymmetric" begin
    A = reshape(collect(1.:8.), 2, 2, 2)
    @test_throws AssertionError issymetric(A)
    @test issymetric([[1.0 2.0]; [2.0 1.0]]) == nothing
  end
  @testset "indexing" begin
    @test indices(2, 3) == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    @test ind2range(2, 3, 5) == 4:5
  end
  context("random generate of symmetric array") do
    srand(40)
    t = randsymarray(4, 2)
    @fact t-transpose(t) --> zeros(4,4)
  end
  context("sizetest") do
    @fact_throws DimensionMismatch sizetest(2,3)
  end
end

# generates symmetric tensors
srand(42)
t = randsymarray(7, 3)
t1 = randsymarray(7, 3)


@testset "Converting" begin
  b = convert(SymmetricTensor, t, 3)
  @testset "Geting blocks of Symmetric Tensors" begin
    @test getblockunsafe(b, (1, 1, 1)) == t[1:3, 1:3, 1:3]
    @test getblock(b, (2, 1, 2)) == t[4:6, 1:3, 4:6]
    @test getblock(b, (2, 1, 1)) == t[4:6, 1:3, 1:3]
  end
  @testset "Geting indices" begin
    @test b[1, 1, 1] == t[1, 1, 1]
    @test b[3, 4, 7] == t[3, 4, 7]
    @test b[3, 3, 4] == t[3, 3, 4]
    @test b[7, 7, 7] == t[7, 7, 7]
    @test b[7, 7, 3] == t[7, 7, 3]
    @test b[6, 4, 2] == t[6, 4, 2]
  end
  @testset "converting from array to SymmetricTensor" begin
    a = reshape(collect(1.:16.), 4, 4)
    @test getblockunsafe(convert(SymmetricTensor, a*a'), (1,1)) ==  [276.0  304.0; 304.0  336.0]
    @test b.frame[1, 1, 1].value ≈ t[1:3, 1:3, 1:3]
    @test b.frame[1, 2, 2].value ≈ t[1:3, 4:6, 4:6]
    @test b.frame[2, 2, 2].value ≈ t[4:6, 4:6, 4:6]
    @test isnull(b.frame[2, 1, 1])
  end
  @testset "Constructor tests" begin
    b1 = convert(SymmetricTensor, t[1:6, 1:6, 1:6], 2)
    @test !(b.sqr)
    @test b1.sqr
    @test b.bls == 3
    @test b.bln == 3
    @test b.dats == 7
  end
end

facts("Random symmetric tensor generation") do
  srand(42)
  s = rand(SymmetricTensor{Float64, 3}, 2)
  a = convert(Array, s)
  t = cat(3, [0.533183 0.454029; 0.454029 0.0176868], [0.454029 0.0176868; 0.0176868 0.172933])
  @fact a --> roughly(t, 1e-5)
  @fact a[:,:,1]-transpose(a[:,:,1]) --> zeros(2,2)
end

facts("Basic operations") do
  b = convert(SymmetricTensor, t)
  b1 = convert(SymmetricTensor, t1)
  @testset "Get super-diagonal" begin
    @test diag(b) ≈ [t[fill(i, ndims(t))...] for i = 1:size(t, 1)]
    @test diag(b1) ≈ [t1[fill(i, ndims(t1))...] for i = 1:size(t1, 1)]
  end
  @testset "Elementwise operations" begin
    @test convert(Array, b + b1) ≈ t + t1
    @test convert(Array, b - b1) ≈ t - t1
    @test convert(Array, b .* b1) ≈ broadcast(*, t, t1)
    @test convert(Array, b ./ b1) ≈ broadcast(/, t, t1)
  end
  @testset "Matrix--scalar operations" begin
    @test convert(Array, b * 2.1) ≈ t * 2.1
    @test convert(Array, b / 2.1) ≈ t / 2.1
    @test convert(Array, b / 2) ≈ t / 2
    @test convert(Array, b + 2.1) ≈ t + 2.1
    @test convert(Array, b - 2.1) ≈ t - 2.1
    @test convert(Array, b + 2) ≈ t + 2
    @test convert(Array, 2 + b) ≈ t + 2
  end
end

@testset "Exceptions" begin
  @testset "Dimensions in operations" begin
    b = convert(SymmetricTensor, t)
    b1 = convert(SymmetricTensor, t, 3)
    b2 = convert(SymmetricTensor, t[1:6, 1:6, 1:6])
    b2 = convert(SymmetricTensor, t[:,:,1])
    @test_throws DimensionMismatch b + b1
    @test_throws MethodError b + b2
    @test_throws UndefVarError b + b3
  end
  @testset "Constructor exceptions" begin
    @test_throws TypeError SymmetricTensor([1.0 2.0]; [3.0 4.0])
    @test_throws DimensionMismatch SymmetricTensor(t[:, :, 1:2])
    b = SymmetricTensor(t).frame
    b1 = copy(b)
    b2 = copy(b)
    b[1,1,1] = reshape(collect(1:8), (2,2,2))
    @test_throws AssertionError SymmetricTensor(b)
    b1[1,2,3] = reshape(collect(1:4), (2,2,1))
    @test_throws AssertionError SymmetricTensor(b1)
    b2[3,2,1] = reshape(collect(1:8), (2,2,2))
    @test_throws AssertionError SymmetricTensor(b2)
    # wrong block size
    @test_throws DimensionMismatch convert(SymmetricTensor, t, 25)
    @test_throws DimensionMismatch convert(SymmetricTensor, t, 0)
  end
end
