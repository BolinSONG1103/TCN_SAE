function [XTrain, YTrain] = makeWindowsTrain(seqs, L, S, RULcap, doOversample)
    % 生成滑窗训练样本；分段过采样末期（小RUL）窗口
    if nargin < 5, doOversample = false; end
    XTrain = {}; YTrain = [];
    for i=1:numel(seqs)
        X = seqs{i};  [~, T] = size(X);
        for t0 = 1:S:(T-L+1)
            tEnd = t0 + L - 1;
            rul  = min(max(T - tEnd,0), RULcap);
            XTrain{end+1,1} = X(:, t0:tEnd); %#ok<AGROW>
            YTrain(end+1,1) = rul;           %#ok<AGROW>

            % —— 分段过采样（轻量）——
            if doOversample
                if     rul < 20, rep = 3;   % 极小RUL更关注
                elseif rul < 40, rep = 2;
                else   rep = 1;
                end
                for rr = 2:rep
                    XTrain{end+1,1} = X(:, t0:tEnd); %#ok<AGROW>
                    YTrain(end+1,1) = rul;           %#ok<AGROW>
                end
            end
        end
    end
    YTrain = single(YTrain);
end