function get_diff{T<:AbstractFloat}(dane::Matrix{T}, r::Int = 4)
    n = size(dane, 2)
    f(t::Vector) = log(mean(exp(t'*dane')))
    t_vec = zeros(Float64, n)
    fgen1 = ForwardDiff.gradient(f)
    fgen2 = hessian(f)
    fgen3 = tensor(f)
    function nthcumgen(gen_funct)
        vecform(x) = vec(gen_funct(x::Vector))
        jacobian(vecform)
    end
    tensor_form(mat::Matrix, size::Int, modes::Int) = reshape(mat,fill(size,modes)...)
    fgen = fgen3
    ret = Any[]
    for modes = 4:r
        fgen = nthcumgen(fgen)
        fn = tensor_form(fgen(t_vec),n, modes)
        push!(ret, fn)
    end
    fgen1(t_vec), fgen2(t_vec), fgen3(t_vec), ret...
end
