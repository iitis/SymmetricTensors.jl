using FactCheck
using SymmetricTensors
using NullableArrays
using Iterators
using Combinatorics

include("test_helpers/generate_data.jl")

rmat, srmat, smseg = generatedata()
rmat2, srmat2, smseg2 = generatedata()
rmat3, srmat3, smseg3 = generatedata(14)
rmat4, srmat4, smseg4 = generatedata(15, 3)

facts("Converting") do
  converttest = convert(SymmetricTensor, srmat[1:6, 1:6, 1:6], 3)
  context("From array to SymmetricTensor") do
    @fact converttest.frame[1,1,1].value --> roughly(srmat[1:3, 1:3, 1:3])
    @fact converttest.frame[1,2,2].value --> roughly(srmat[1:3, 4:6, 4:6])
    @fact converttest.frame[2,2,2].value --> roughly(srmat[4:6, 4:6, 4:6])
    @fact isnull(converttest.frame[2,1,1]) --> true
  end

  context("From SymmetricTensor to array") do
    @fact convert(Array, smseg) --> roughly(srmat)
  end
end

facts("Basic operations") do
  context("Matrix--matrix elementwise operations") do
    @fact convert(Array,smseg+smseg2) --> roughly(srmat+srmat2)
    @fact convert(Array,smseg-smseg2) --> roughly(srmat-srmat2)
    @fact convert(Array,smseg.*smseg2) --> roughly(srmat.*srmat2)
    @fact convert(Array,smseg./smseg2) --> roughly(srmat./srmat2)
  end

  context("Matrix--scalar operations") do
    @fact convert(Array,smseg*2.1) -->roughly(srmat*2.1)
    @fact convert(Array,smseg/2.1) -->roughly(srmat/2.1)
    @fact convert(Array,smseg/2) -->roughly(srmat/2)
    @fact convert(Array,smseg+2.1) -->roughly(srmat+2.1)
    @fact convert(Array,smseg-2.1) -->roughly(srmat-2.1)
    @fact convert(Array,smseg+2) -->roughly(srmat+2)
  end
end

facts("Exceptions") do
  context("Dimensions in operations") do
    @fact_throws DimensionMismatch, smseg+smseg3
    @fact_throws DimensionMismatch, smseg.*smseg3
    @fact_throws DimensionMismatch, smseg+smseg4
    @fact_throws DimensionMismatch, smseg.*smseg4
    @fact_throws DimensionMismatch, smseg+convert(SymmetricTensor, srmat[1,:,:], 4)
  end

  context("Constructor exceptions") do
    @fact_throws AssertionError, SymmetricTensor(rmat)
    @fact_throws AssertionError, SymmetricTensor(srmat[:, :, 1:2])
    @fact_throws AssertionError, SymmetricTensor(create_except(rand(4,4,4)))
    @fact_throws AssertionError, SymmetricTensor(create_except(srmat, false, true))
    @fact_throws AssertionError, SymmetricTensor(create_except(srmat, true))
    # wrong block size
    @fact_throws DimensionMismatch, convert(SymmetricTensor, srmat,  25)
    @fact_throws DimensionMismatch, convert(SymmetricTensor, srmat,  0)
  end
end

facts("Helper functions") do
  context("unfold") do
    A = reshape(collect(1:8), 2, 2, 2)
    @fact unfold(A, 1) --> [[1 3 5 7]; [2 4 6 8]]
    @fact unfold(A, 2) --> [[1 2 5 6]; [3 4 7 8]]
    @fact unfold(A, 3) --> [[1 2 3 4]; [5 6 7 8]]
  end
end
