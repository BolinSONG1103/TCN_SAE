function [XTest, YTest] = lastWindowForTest(seqs, rulTest, L, RULcap)
    % 每台只取最后一个窗口；标签为官方 RUL（截断）
    N = numel(seqs);
    XTest = cell(N,1);
    Y = single(rulTest(:));
    if numel(Y)~=N, error('RUL 行数与 test units 不一致'); end
    for i=1:N
        X = seqs{i}; [~,T] = size(X);
        if T < L
            pad = repmat(X(:,1),1,L-T); X = [pad, X];
        end
        XTest{i} = X(:, end-L+1:end);
    end
    YTest = min(Y, RULcap);
end