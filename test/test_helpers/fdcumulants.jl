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
