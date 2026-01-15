function seqs = groupByUnit(A, featureIdx)
    % 按 unit ID 聚合成序列，[F x T]
    units = unique(A(:,1));
    seqs = cell(numel(units),1);
    for i = 1:numel(units)
        tmp = A(A(:,1)==units(i), :);
        X   = tmp(:, featureIdx).';
        seqs{i} = X;
    end
end