using Test
using SymmetricTensors
using Random
using Combinatorics

import SymmetricTensors: ind2range, pyramidindices, issymetric, sizetest,
getblock, getblockunsafe, randsymarray, fixpointperms, randblock, setindexunsafe!


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
    @test pyramidindices(2, 3) == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    @test ind2range(2, 3, 5) == 4:5
  end
  @testset "random generate of symmetric array" begin
    Random.seed!(40)
    t = randsymarray(4, 2)
    @test (t-transpose(t)) == zeros(4,4)
  end
  @testset "sizetest" begin
    @test_throws DimensionMismatch sizetest(2,3)
  end
end

# generates symmetric tensors
Random.seed!(42)
t = randsymarray(7, 3)
t1 = randsymarray(7, 3)


@testset "Converting" begin
  b = SymmetricTensor(t, 3)
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
    @test getblockunsafe(SymmetricTensor(a*a'), (1,1)) ==  [276.0  304.0; 304.0  336.0]
    @test b.frame[1, 1, 1] ≈ t[1:3, 1:3, 1:3]
    @test b.frame[1, 2, 2] ≈ t[1:3, 4:6, 4:6]
    @test b.frame[2, 2, 2] ≈ t[4:6, 4:6, 4:6]
    @test b.frame[2, 1, 1] == nothing
  end
  @testset "Constructor tests" begin
    b1 = SymmetricTensor(t[1:6, 1:6, 1:6], 2)
    @test !(b.sqr)
    @test b1.sqr
    @test b.bls == 3
    @test b.bln == 3
    @test b.dats == 7
  end
end

@testset "Random symmetric tensor generation" begin
  @test fixpointperms((1,2,3,3)) == [[1,2,3,4],[1,2,4,3]]
  @test fixpointperms((1,2,3,4)) == [[1,2,3,4]]
  Random.seed!(42)
  aa = Array(rand(SymmetricTensor{Float64, 4}, 3))
  tt = cat([0.533183 0.454029; 0.454029 0.0176868], [0.454029 0.0176868; 0.0176868 0.172933], dims = 3)
  #@test aa ≈ tt atol=1.0e-5
  issymetric(aa)
  @test aa == permutedims(aa, (2,1,3,4)) == permutedims(aa, (2,3,1,4)) == permutedims(aa, (4,3,1,2))
  @test aa[:,:,1,1] == transpose(aa[:,:,1,1])
  Random.seed!(42)
  @test randblock(Float64, (2,2), (1,1)) ≈ [0.533183  0.454029; 0.454029  0.0176868] atol=1.0e-5
  @test randblock(Float64, (2,2), (1,2)) ≈ [0.172933  0.973566; 0.958926  0.30387] atol=1.0e-5
end

@testset "Seting value" begin
    @testset "Unsafe" begin
      Random.seed!(42)
      x = rand(SymmetricTensor{Float64, 4}, 7)
      y = rand(SymmetricTensor{Float64, 3}, 7)
      setindexunsafe!(x, 10000., 1,1,2,2)
      setindexunsafe!(x, 100., 4,4,4,4)
      setindexunsafe!(x, 200., 3,5,6,7)
      @test prod(map(i -> x[(1,1,2,2)[i]...], collect(permutations(1:4))) .== 10000.)
      @test prod(map(i -> x[(3,5,6,7)[i]...], collect(permutations(1:4))) .== 200.)
      @test x[4,4,4,4] == 100.
      issymetric(Array(x))
      setindexunsafe!(y, 10000., 1,2,3)
      setindexunsafe!(y, 100., 7,7,7)
      @test prod(map(i -> y[(1,2,3)[i]...], collect(permutations(1:3))) .== 10000.)
      @test y[7,7,7] == 100.
      issymetric(Array(y))
    end
    @testset "safe" begin
      x = rand(SymmetricTensor{Float64, 4}, 7)
      x[7,4,3,1] = 20.
      @test prod(map(i -> x[(1,3,4,7)[i]...], collect(permutations(1:4))) .== 20.)
      issymetric(Array(x))
    end
end

@testset "Basic operations" begin
  b = SymmetricTensor(t)
  b1 = SymmetricTensor(t1)
  @testset "Get super-diagonal" begin
    @test diag(b) ≈ [t[fill(i, ndims(t))...,] for i = 1:size(t, 1)]
    @test diag(b1) ≈ [t1[fill(i, ndims(t1))...,] for i = 1:size(t1, 1)]
  end
  @testset "Elementwise operations" begin
    @test Array(b + b1) ≈ t + t1
    @test Array(b - b1) ≈ t - t1
  end
  @testset "Matrix--scalar operations" begin
    @test Array(b * 2.1) ≈ t * 2.1
    @test Array(b / 2.1) ≈ t / 2.1
    @test Array(b / 2) ≈ t / 2
    @test Array(b + 2.1) ≈ t .+ 2.1
    @test Array(b - 2.1) ≈ t .- 2.1
    @test Array(b + 2) ≈ t .+ 2
    @test Array(2 + b) ≈ t .+ 2
  end
end

@testset "Exceptions" begin
  @testset "Dimensions in operations" begin
    b = SymmetricTensor(t)
    b1 = SymmetricTensor(t, 4)
    b4 = SymmetricTensor(t[1:4, 1:4, 1:4])
    b5 = SymmetricTensor(t[1:3, 1:3, 1:3])
    b2 = SymmetricTensor(t[:,:,1])
    @test_throws DimensionMismatch b + b1
    @test_throws DimensionMismatch b4 + b5
    @test_throws MethodError b + b2
    @test_throws UndefVarError b + b3
  end
  @testset "Constructor exceptions" begin
    @test_throws MethodError SymmetricTensor([1.0 2.0], [3.0 4.0])
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
    @test_throws DimensionMismatch SymmetricTensor(t, 25)
    @test_throws DimensionMismatch SymmetricTensor(t, 0)
  end
end
