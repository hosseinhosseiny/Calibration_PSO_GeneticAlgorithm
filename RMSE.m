function [rmse] = RMSE(n,net,wse_mes)
n_sc = (n - 0.001)/(0.03 - 0.001);
ws_sc = predict(net,n_sc);
ws = ws_sc *(449.494 - 447.103) + 447.103;
rmse = sqrt(mean((ws-wse_mes).^2));

end
%particle(i).Position
