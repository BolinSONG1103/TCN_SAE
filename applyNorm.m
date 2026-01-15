function outSeqs = applyNorm(seqs, mu, sig)
    % 应用标准化
    outSeqs = cell(size(seqs));
    for i=1:numel(seqs)
        X = seqs{i};
        outSeqs{i} = (X - mu) ./ sig;
    end
end