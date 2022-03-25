function [W1, W2] = multiClas(W1, W2, X, D, maxEpocas)
    global m
    a = 0.1;
    for epoca = 1: maxEpocas
        for i=1:m
            x = X(i, :)';
            d = D(i, :)';
            
            % capa oculta
            v1 = W1 * x;
            y1 = sigmoide(v1);

            % capa final
            v = W2 * y1;
            y = softmax(v);

            % calculamos el error
            e = d - y;
            delta = e;

            error(i) = sum(abs(e));

            % retropropagamos el error
            e1 = W2' * delta;
            delta1 = y1 .* (1 - y1) .* e1;

            % actualizamos los pesos
            dW1 = a * delta1 * x';
            W1 = W1 + dW1;

            dW2 = a * delta * y1';
            W2 = W2 + dW2;
        end
        convergencia(epoca) = sum(error);
    end
    figure
    plot(convergencia)
end