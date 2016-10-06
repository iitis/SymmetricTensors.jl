using FactCheck
using SymmetricTensors
using Distributions
using NullableArrays
using Iterators

include("test_helpers/generate_data.jl")
include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")


rmat, srmat, smseg = generatedata()
rmat2, srmat2, smseg2 = generatedata()
rmat3, srmat3, smseg3 = generatedata(14)
rmat4, srmat4, smseg4 = generatedata(15, 3)

gaus_dat =  [[-0.88626   0.279571];
            [-0.704774  0.131896]]


facts("Converting") do
  converttest = convert(SymmetricTensor, srmat[1:6, 1:6, 1:6])
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
    # to may blocks
    @fact_throws DimensionMismatch, convert(SymmetricTensor, srmat,  8)
  end
end

facts("Helper functions") do
  context("center") do
    @fact sum(abs(mean(center(rmat[1:3, 1:10]), 1))) --> roughly(0, 1e-15)
  end
  context("unfold") do
    A = reshape(collect(1:8), 2, 2, 2)
    @fact unfold(A, 1) --> [[1 3 5 7]; [2 4 6 8]]
    @fact unfold(A, 2) --> [[1 2 5 6]; [3 4 7 8]]
    @fact unfold(A, 3) --> [[1 2 3 4]; [5 6 7 8]]
  end
end

data = clcopulagen(10, 4)
facts("Moments") do
  context("3") do
    @fact convert(Array, momentbs(data, 3, 2)) --> roughly(moment_n(data, 3))
  end

  context("4") do
    @fact convert(Array, momentbs(data, 4, 2)) --> roughly(moment_n(data, 4))
  end
end

facts("Comulants vs naive implementation") do
  cn = [naivecumulant(data, i) for i = 2:6]
  context("Square blocks") do
    c2, c3, c4, c5, c6 = cumulants(6, data, 2)
    @fact convert(Array, c2) --> roughly(cn[1])
    @fact convert(Array, c3) --> roughly(cn[2])
    @fact convert(Array, c4) --> roughly(cn[3])
    @fact convert(Array, c5) --> roughly(cn[4])
    @fact convert(Array, c6) --> roughly(cn[5])
  end

  context("Non-square blocks") do
    c2, c3, c4, c5, c6 = cumulants(6, data[:, 1:3], 2)
    @fact convert(Array, c2) --> roughly(cn[1][fill(1:3, 2)...])
    @fact convert(Array, c3) --> roughly(cn[2][fill(1:3, 3)...])
    @fact convert(Array, c4) --> roughly(cn[3][fill(1:3, 4)...])
    @fact convert(Array, c5) --> roughly(cn[4][fill(1:3, 5)...])
    @fact convert(Array, c6) --> roughly(cn[5][fill(1:3, 6)...])
  end
end

facts("test semi-naive against gaussian") do
  cg = snaivecumulant(gaus_dat, 8)
  @fact cg["c2"] --> roughly(naivecumulant(gaus_dat, 2))
  @fact cg["c3"] --> roughly(zeros(Float64, 2,2,2))
  @fact cg["c4"] --> roughly(zeros(Float64, 2,2,2,2), 1e-3)
  @fact cg["c5"] --> roughly(zeros(Float64, 2,2,2,2,2))
  @fact cg["c6"] --> roughly(zeros(Float64, 2,2,2,2,2,2), 1e-4)
  @fact cg["c7"] --> roughly(zeros(Float64, 2,2,2,2,2,2,2))
  @fact cg["c8"] --> roughly(zeros(Float64, 2,2,2,2,2,2,2,2), 1e-5)
end

facts("Cumulants vs semi-naive non-square") do
  c2, c3, c4, c5, c6, c7, c8 = cumulants(8, data[:, 1:2], 2)
  cnn = snaivecumulant(data[:, 1:2], 8)
  @fact convert(Array, c2) --> roughly(cnn["c2"])
  @fact convert(Array, c3) --> roughly(cnn["c3"])
  @fact convert(Array, c4) --> roughly(cnn["c4"])
  @fact convert(Array, c5) --> roughly(cnn["c5"])
  @fact convert(Array, c6) --> roughly(cnn["c6"])
  @fact convert(Array, c7) --> roughly(cnn["c7"])
  @fact convert(Array, c8) --> roughly(cnn["c8"])
end
