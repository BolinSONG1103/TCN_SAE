function YPred = predictLastK(seqs, rulTest, net, L, RULcap, K)
    % 对最后 K 个窗口做线性加权平均（越靠末端权重越大）
    N = numel(seqs);  YPred = zeros(N,1,'single');
    w = linspace(0.5, 1, K)';    % 末端权重大
    w = w / sum(w);
    for i=1:N
        X = seqs{i}; [~,T] = size(X);
        if T < L, pad=repmat(X(:,1),1,L-T); X=[pad,X]; T=L; end
        Kuse = min(K, T-L+1);
        vals = zeros(Kuse,1);
        for k = 1:Kuse
            tEnd = T - (Kuse-k);
            Xi = X(:, tEnd-L+1:tEnd);
            vals(k) = predict(net, {Xi});
        end
        YPred(i) = sum(vals .* w(end-Kuse+1:end));
    end
end