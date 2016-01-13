using DataFrames

function read_financial(file)
    pkobp = readtable(file)
    lenght=size(pkobp[:_CLOSE_], 1)
    return_matrix = zeros(lenght-1, 20)

    share_names = DataFrame(share_name = AbstractString[])
    files = readdir()
    k = 0
    for i = 1:size(files,1)
        if contains(string(files[i]), ".mst")
            k = k+1
            data = readtable(files[i])
            data = data[data[:_DTYYYYMMDD_].>=pkobp[:_DTYYYYMMDD_][1],:]
            l = 1
            for j = 1:lenght-1
                if data[:_DTYYYYMMDD_][l]==pkobp[:_DTYYYYMMDD_][j]
                    return_matrix[j,k] = 100*diff(data[:_CLOSE_])[l]/data[:_CLOSE_][l]
                    l = l+1
                else
                    return_matrix[j,k] = 0
                end

            end
            push!(share_names, [string(files[i])])

        end
    end
    n = size(share_names,1)
    return_matrix, n, share_names
end
