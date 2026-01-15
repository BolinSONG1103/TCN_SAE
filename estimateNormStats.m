function [mu, sig] = estimateNormStats(seqs)
    % 汇总训练序列做 z-score 统计
    allcat = [];
    for i=1:numel(seqs), allcat = [allcat, seqs{i}]; end %#ok<AGROW>
    mu  = mean(allcat, 2);
    sig = std(allcat, 0, 2) + 1e-8;
end