datos = load('dataset_multiclassOK.txt');

[m,n] = size(datos);

X = datos(:,1:n-1);
Y = datos(:, n);

X = zscore(X);

noClases = numel(unique(Y));
D = zeros(m, noClases);

for i=1: m
    D(i, Y(i)) = 1;
end

W1 = 2 * rand(10, n-1) - 1;
W2 = 2 * rand(noClases, 10) - 1;

maxEpocas = 200;
