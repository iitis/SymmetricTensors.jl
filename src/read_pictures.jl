using NPZ

function read_hyperspectral(file, slise)
    var = npzread(file)
    tab = convert(Array{Float64},var)[:,:,1:slise:end]
    n,m,l = size(tab)
    reshape(tab, n*m,l), l
end  
