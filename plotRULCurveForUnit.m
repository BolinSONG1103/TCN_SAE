function plotRULCurveForUnit(testSeqs, rulTest, unitIdx, net, L, RULcap)
% 可视化单台发动机随时间的 RUL 曲线（蓝=预测，黑=理论直线）
    assert(unitIdx>=1 && unitIdx<=numel(testSeqs), 'unitIdx 超界');

    X = testSeqs{unitIdx}; [~, T] = size(X);
    if T < L
        pad = repmat(X(:,1), 1, L - T); X = [pad, X]; T = L;
    end

    % 逐窗口滚动预测
    numSteps = T - L + 1;
    Ypred = zeros(numSteps,1);
    for i = 1:numSteps
        Xi = X(:, i:i+L-1);
        Ypred(i) = predict(net, {Xi});
    end
    Ypred = movmean(Ypred, 3);

    % 理论真值曲线（以最后真值为锚点的线性下降）
    rul_last = min(rulTest(unitIdx), RULcap);
    tvec  = (L:T)';
    Ytrue = min(rul_last + (T - tvec), RULcap);

    figure; hold on; grid on;
    plot(tvec, Ytrue, 'k-', 'LineWidth', 2, 'DisplayName','True RUL');
    plot(tvec, Ypred, 'b-', 'LineWidth', 2, 'DisplayName','Predicted RUL');
    xlabel('Cycle (window end time)'); ylabel('RUL');
    title(sprintf('Unit #%d  RUL curve (L=%d)', unitIdx, L));
    legend('Location','northeast');

    mmax = max([max(Ytrue), max(Ypred)]);
    rmse = sqrt(mean((Ypred - Ytrue).^2));
    mae  = mean(abs(Ypred - Ytrue));
    text(tvec(1), 0.95*mmax, sprintf('RMSE=%.2f, MAE=%.2f', rmse, mae), ...
        'FontSize',10, 'BackgroundColor','w');
end