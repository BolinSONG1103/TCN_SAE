%% ============================ 本地函数 ============================

function [trainSeqs, testSeqs, rulTest, featureNames] = readFD001(root, fd)
    % 读取FD001数据；选择信息量较高的传感器列
    ftrain = fullfile(root, ['train_' fd '.txt']);
    ftest  = fullfile(root, ['test_'  fd '.txt']);
    frul   = fullfile(root, ['RUL_'   fd '.txt']);
    if ~exist(ftrain,'file')||~exist(ftest,'file')||~exist(frul,'file')
        error('未找到数据文件：train_%s / test_%s / RUL_%s',fd,fd,fd);
    end
    Atrain = readmatrix(ftrain);
    Atest  = readmatrix(ftest);
    rulTest = readmatrix(frul);

    % 去掉全NaN列
    Atrain = Atrain(:, any(~isnan(Atrain),1));
    Atest  = Atest(:,  any(~isnan(Atest),1));

    % 1..unit, 2..cycle, 3..5 settings, 6..26 sensors s1..s21
    keepSensors = [2,3,4,7,8,9,11,12,13,14,15,17,20,21];  % 经验有效
    sensorCols  = 5 + keepSensors;
    featureIdx  = [3 4 5 sensorCols];                    % settings + 上述传感器
    featureNames = "x"+string(1:numel(featureIdx));

    trainSeqs = groupByUnit(Atrain, featureIdx);
    testSeqs  = groupByUnit(Atest,  featureIdx);
end