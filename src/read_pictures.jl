using NPZ

read_hyperspectral(file, step=1) = npzread(file)[:,1:step:end]
