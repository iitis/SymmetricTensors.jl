#old_code_use for comparison

function funct(matrix, t)
    log(mean(exp(1im*t'*matrix')))
end


function differenciate(matrix, vector, vector1, delta)
    (funct(matrix, vector)-funct(matrix, vector1))/(2*delta)
end

function cumulant_old3(matrix, delta, index, index1, index2)
    t = zeros(n)
    t1 = zeros(n)
    t2 = zeros(n)
    t[index] = delta
    t1[index1] = delta
    t2[index2] = delta
    1im*((differenciate(matrix, t+t1+t2, -t+t1+t2, delta)
    -differenciate(matrix, t-t1+t2, -t-t1+t2, delta))/(2*delta)
    -(differenciate(matrix, t+t1-t2, -t+t1-t2, delta)-
    differenciate(matrix, t-t1-t2, -t-t1-t2, delta))/(2*delta))/(2*delta)
end

get_cumulant_oldT3(data_matrix) = Float64[real(cumulant_old3(data_matrix, 0.02,i,j,k)) for i = 1:n, j = 1:n, k = 1:n]


function cumulant_old4(matrix, delta, index, index1, index2, index3)
    t = zeros(n)
    t1 = zeros(n)
    t2 = zeros(n)
    t3 = zeros(n)
    t[index] = delta
    t1[index1] = delta
    t2[index2] = delta
    t3[index3] = delta
    (((differenciate(matrix, t+t1+t2+t3, -t+t1+t2+t3, delta)
    -differenciate(matrix, t-t1+t2+t3, -t-t1+t2+t3, delta))/(2*delta)
    -(differenciate(matrix, t+t1-t2+t3, -t+t1-t2+t3, delta)-
    differenciate(matrix, t-t1-t2+t3, -t-t1-t2+t3, delta))/(2*delta))/(2*delta)-
    ((differenciate(matrix, t+t1+t2-t3, -t+t1+t2-t3, delta)
    -differenciate(matrix, t-t1+t2-t3, -t-t1+t2-t3, delta))/(2*delta)
    -(differenciate(matrix, t+t1-t2-t3, -t+t1-t2-t3, delta)-
    differenciate(matrix, t-t1-t2-t3, -t-t1-t2-t3, delta))/(2*delta))/(2*delta))/(2*delta)

end

get_cumulant_oldT4(data_matrix) =
Float64[real(cumulant_old4(data_matrix, 0.02,i,j,k,l)) for i = 1:n, j = 1:n, k = 1:n, l = 1:n];
