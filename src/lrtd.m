
path(path, '/home/krzysztof/MATLAB/src/tensorAlgs')
path(path, '/home/krzysztof/MATLAB/src/tensor_toolbox_2.6')
path(path, '/home/krzysztof/MATLAB/src/grassClasses')
load('/home/krzysztof/Dokumenty/badania_iitis/tensors_symetric/tensor calculations/pictures_tensor/cumulants.mat')
%zcTest()

A = tensor(C4);
tolALS = 5e-13;
initIt = 20;
k = 3;

[U,S,V]= svd(double(tam(A,1)));
X = U(:,1:k);
[solInit, fInit, nnInit] = algALS(A,{X,X,X,X},[k,k,k,k],initIt,tolALS);








