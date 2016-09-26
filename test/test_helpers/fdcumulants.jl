function cumulantsfd{T<:AbstractFloat}(dane::Matrix{T}, r::Int = 4)
  fgen2(p) = ForwardDiff.hessian(t -> log(mean(exp(t'*dane'))), p)
  nthcumgen(gen_funct) = ForwardDiff.jacobian(x -> vec(gen_funct(x)))
  n = size(dane, 2)
  t_vec = zeros(Float64, n)
  tensor_form(mat::Matrix, s::Int, m::Int) = reshape(mat,fill(s,m)...)
  ret = Any[]
  push!(ret, fgen2(t_vec))
  fgen = fgen2
  for modes = 3:r
      fgen = nthcumgen(fgen)
      push!(ret, tensor_form(fgen(t_vec),n, modes))
  end
  ret
end


using ForwardDiff
dane = randn(2,2)
#cumulantsfd(dane,2)[1]
kappa(t::Vector) = log(mean(exp(t'*dane')))
fgen2(p::Vector) = ForwardDiff.hessian(kappa, p)
function nthcumgen(gen_funct, x::Vector)
    f(x::Vector) = vec(gen_funct(x))
    ForwardDiff.jacobian(f, x)
end
n = size(dane, 2)
t_vec = zeros(Float64, n)
#tensor_form(mat::Matrix, s::Int, m::Int) = reshape(mat,fill(s,m)...)
#nthcumgen(fgen2, t_vec)
#f(x::Vector) = fgen2(x::Vector)
#f(x::Vector) = nthcumgen(f, x)
#f(x::Vector) = nthcumgen(f, x)
f1(x::Vector) = nthcumgen(fgen2, x)
f2(x::Vector) = nthcumgen(f1, x)
f3(x::Vector) = nthcumgen(f2, x)
nthcumgen(f3, t_vec)
