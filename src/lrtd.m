function [U3,U4] = lrtd(k3, k4)
path(path, '/home/krzysztof/MATLAB/src/tensorAlgs')
path(path, '/home/krzysztof/MATLAB/src/tensor_toolbox_2.6')
path(path, '/home/krzysztof/MATLAB/src/grassClasses')
pa = '/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/';
load(strcat(pa, 'cumulants.mat'));

A4 = tensor(C4);
A3 = tensor(C3);
tolALS = 5e-13;
It = 20;

[sol_4, f_4, nn_4] = algALS(A4,{X4,X4,X4,X4},[k4,k4,k4,k4],It,tolALS);
[sol_3, f_3, nn_3] = algALS(A3,{X3,X3,X3},[k3,k3,k3],It,tolALS);

U3 = sol_3{1,1};
U4 = sol_4{1,1};
% save(strcat(pa, 'ALS.mat'), 'U3', 'U4');
end


 