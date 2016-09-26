function permute(array::Vector{Int})
    a = Vector{Vector{Int}}[]
    n = size(array, 1)
    for p in partitions(array)
        add = true
        for k in p
            if size(k,1) in[1, n]
                add = false
            end
        end
        if add
            push!(a, p)
        end
    end
    return a
end

macro per(a, b ,i)
    eval = quote
        if size($i,1) == 3
            I = [$i[1], $i[2], $i[3]]
        elseif size($i,1) == 4
            I = [$i[1], $i[2], $i[3], $i[4]]
        elseif size($i,1) == 5
            I = [$i[1], $i[2], $i[3], $i[4], $i[5]]
        elseif size($i,1) == 6
            I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6]]
        elseif size($i,1) == 7
            I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7]]
        elseif size($i,1) == 8
            I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7], $i[8]]
        elseif size($i,1) == 9
            I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7], $i[8], $i[9]]
        elseif size($i,1) == 10
            I = [$i[1], $i[2], $i[3], $i[4], $i[5], $i[6], $i[7], $i[8], $i[9], $i[10]]
        end
        for per in collect(permutations(I))
            $a[per...] = $b
        end
    end
    return (eval)
    end


function permutations_output!{T<:AbstractFloat, N}(m4::AbstractArray{T, N}, a::T, list::Vector{Int})
    @per(m4, a, list)
end


function moment_element!{T<:AbstractFloat, N}(moment::Array{T, N}, indices::Vector{Int}, data::Matrix{T})
    multiple = 1
    for i in indices
        multiple  = multiple.*data[:,i]
    end
    permutations_output!(moment, mean(multiple), indices)
end

function moment3{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,3)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m
        moment_element!(moment, [i1,i2,i3], data)
    end
    return Array(moment)
end


function moment4{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,4)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m
        moment_element!(moment, [i1,i2,i3,i4], data)
    end
    return Array(moment)
end


function moment5{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,5)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m
        moment_element!(moment, [i1,i2,i3,i4,i5], data)
    end
    return Array(moment)
end


function moment6{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,6)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m
        moment_element!(moment, [i1,i2,i3,i4,i5,i6], data)
    end
    return Array(moment)
end

function moment7{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,7)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m
        moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7], data)
    end
    return Array(moment)
end

function moment8{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,8)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m
        moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7,i8], data)
    end
    return Array(moment)
end


function moment9{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,9)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m
        moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7,i8,i9], data)
    end
    return Array(moment)
end

function moment10{T<:AbstractFloat}(data::Matrix{T})
    m = size(data,2)
    moment = zeros(fill(m,10)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m, i10 = i9:m
        moment_element!(moment, [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10], data)
    end
    return Array(moment)
end

function moment_n{T<:AbstractFloat}(data::Matrix{T}, n::Int)
    m = size(data,2)
    moment = zeros(fill(m,n)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m, i10 = i9:m
        indices = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10][1:n]
        moment_element!(moment, indices, data)
    end
    return Array(moment)
end


function calculate_el{T<:AbstractString}(c::Dict{T ,Any}, list::Vector{Int})
    a = permute(list)
    w = 0
    for k = 1:size(a,1)
        r = 1
        for el in a[k]
            r*= c["c"*"$(size(el,1))"][el...]
        end
        w += r
    end
    return w
end

function product4{T<:AbstractString}(c::Dict{T ,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,4)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m
        indices = [i1,i2,i3,i4]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function product5{T<:AbstractString}(c::Dict{T ,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,5)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m
        indices = [i1,i2,i3,i4, i5]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function product6{T<:AbstractString}(c::Dict{T ,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,6)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m
        indices = [i1,i2,i3,i4, i5, i6]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function product7{T<:AbstractString}(c::Dict{T ,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,7)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m
        indices = [i1,i2,i3,i4, i5, i6, i7]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function product8{T<:AbstractString}(c::Dict{T ,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,8)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m
        indices = [i1,i2,i3,i4, i5, i6, i7, i8]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function product9{T<:AbstractString}(c::Dict{T ,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,9)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m
        indices = [i1,i2,i3,i4, i5, i6, i7, i8, i9]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function product10(c::Dict{AbstractString,Any})
    m = size(c["c2"], 2)
    m4 = zeros(fill(m,10)...)
    for i1 = 1:m, i2 = i1:m, i3 = i2:m, i4 = i3:m, i5 = i4:m, i6 = i5:m, i7 = i6:m, i8 = i7:m, i9 = i8:m, i10 = i9:m
        indices = [i1,i2,i3,i4, i5, i6, i7, i8, i9, i10]
        a = calculate_el(c, indices)
        permutations_output!(m4, a, indices)
    end
    return Array(m4)
end

function snaivecumulant{T<:AbstractFloat}(data::Matrix{T}, n::Int)
    data = center(data);
    if VERSION >= v"0.5.0-dev+1204"
      c2 = Base.covm(data, 0, 1, false)
    elseif VERSION < v"0.5.0-dev+1204"
      c2 = Base.covm(data, 0; corrected = false)
    end
    c3 = moment3(data)
    cumulants = Dict("c2" => c2, "c3" => c3);
    if n == 3
        return cumulants
    end
    c4 = moment4(data) - product4(cumulants)
    cumulants = merge(cumulants, Dict("c4" => c4));
    if n == 4
        return cumulants
    end
    c5 = moment5(data)-product5(cumulants)
    cumulants = merge(cumulants, Dict("c5" => c5))
    if n == 5
        return cumulants
    end
    c6 = moment6(data)-product6(cumulants)
    cumulants = merge(cumulants, Dict("c6" => c6))
    if n == 6
        return cumulants
    end
    c7 = moment7(data)-product7(cumulants)
    cumulants = merge(cumulants, Dict("c7" => c7))
    if n == 7
        return cumulants
    end
    c8 = moment8(data)-product8(cumulants)
    cumulants = merge(cumulants, Dict("c8" => c8))
    if n == 8
        return cumulants
    end
    c9 = moment9(data)-product9(cumulants)
    cumulants = merge(cumulants, Dict("c9" => c9))
    if n == 9
        return cumulants
    end
    c10 = moment10(data)-product10(cumulants)
    cumulants = merge(cumulants, Dict("c10" => c10))
    return cumulants
end
