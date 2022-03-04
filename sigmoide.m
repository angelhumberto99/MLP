function y1 = sigmoide(v1)
    y1 = 1 ./ (1 + exp(-v1));
end