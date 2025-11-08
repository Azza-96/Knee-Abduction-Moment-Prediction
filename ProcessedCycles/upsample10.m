
function y_up = upsample10(x)
    y_up = interp1(1:size(x,1), x, linspace(1, size(x,1), size(x,1)*10), 'linear');
end