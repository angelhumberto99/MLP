clear all
global m
datos = load('dataset_multiclassOK.txt');

[m,n] = size(datos);

X = datos(:,1:n-1);
Y = datos(:, n);

% normalizamos los datos
media = mean(X);
sigma = std(X);
X = zscore(X);

% obtiene el número de clases unicas
noClases = numel(unique(Y));

D = zeros(m, noClases);
for i=1: m
    D(i, Y(i)) = 1;
end

W1 = 2 * rand(10, n-1) - 1;
W2 = 2 * rand(noClases, 10) - 1;

maxEpocas = 200;

[W1, W2] = multiClas(W1,W2,X,D,maxEpocas);

for i=1:m
    x = X(i, :)';
    
    % capa oculta
    v1 = W1 * x;
    y1 = sigmoide(v1);

    % capa final
    v = W2 * y1;
    y(i, :) = softmax(v);
end
y = y > 0.8;
for i=1:m
    for j=1:noClases
        if y(i,j) == 1
            class(i,1) = j;
        end
    end
end
contador= 0;
for i=1:m
    if class(i) == Y(i)
        contador = contador + 1;
    end
end
error = m - contador;
datoEntrada = [47,2,3,6500,44300];
datoEntradaNorm = (datoEntrada - media)/sigma;

x = datoEntradaNorm';
    
% capa oculta
v1 = W1 * x;
y1 = sigmoide(v1);

% capa final
v = W2 * y1;
yDatoEntrada = softmax(v)

