using FactCheck
using Base.Test
using SymmetricTensors
using Distributions
using ForwardDiff
using NullableArrays
using Iterators
# include("Test.jl")

srand(42)

symmetrise(m::Matrix) = m*m.'

function generatedata(n::Int = 15, seg::Int = 4, l::Int = 1000)
    rmat = randn(n,n)
    srmat = symmetrise(rmat)
    rmat, srmat, convert(SymmetricTensor, srmat, seg), rand(l,n), bitrand(n,n), (im*rmat + rmat)*(im*rmat + rmat)'
 end

rmat, srmat, smseg, data, boolean, comlx = generatedata()
rmat2, srmat2, smseg2 = generatedata()

facts("Converting") do
  converttest = convert(SymmetricTensor, srmat[1:6, 1:6])
  context("From array to SymmetricTensor") do
    @fact converttest.frame[1,1].value --> roughly(srmat[1:3, 1:3])
    @fact converttest.frame[1,2].value --> roughly(srmat[1:3, 4:6])
    @fact converttest.frame[2,2].value --> roughly(srmat[4:6, 4:6])
    @fact isnull(converttest.frame[2,1]) --> true
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
