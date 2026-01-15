clc; clear; close all;

%% -------------------- 路径与超参数 --------------------
dataRoot   = 'C:\Users\25691\OneDrive\Desktop\CMAPSSData';  % ← 修改为你的数据路径
fd         = 'FD001';

% 滑窗/标签
L       = 50;     % 窗口长度
S       = 3;      % 滑动步长（S=3：速度与样本量的平衡）
RULcap  = 130;    % RUL 上限截断

% SAE / TCN
D_embed   = 64;   % SAE 输出/TCN通道数
E1 = 128; E2 = 64;
drop      = 0.1;
numBlocks = 4;    % 残差块数（浅 → 更快）
ksize     = 5;    % 卷积核（5 比 3 的感受野更大，仍较快）

% 训练
miniBatchSize = 256;     % 显存不够可改回 128
maxEpochs     = 40;      % 配合早停即可
learnRate     = 1e-3;    % Adam + LN 收敛较快
rng(2025);

%% -------------------- 读数 & 预处理 --------------------
[trainSeqs, testSeqs, rulTest, featureNames] = readFD001(dataRoot, fd);
F = size(trainSeqs{1},1);  % 特征维度（筛选后）

% 统一统计量（训练集所有序列拼接）
[mu, sig] = estimateNormStats(trainSeqs);

% 标准化
trainSeqs = applyNorm(trainSeqs, mu, sig);
testSeqs  = applyNorm(testSeqs,  mu, sig);

%% --- 按 unit 划分 90/10 训练/验证（避免时间泄漏） ---
Nunits = numel(trainSeqs);
idxVal = (Nunits-9):Nunits;      % 最后10台做验证
idxTr  = 1:(Nunits-10);

% 训练端启用"小 RUL 分段过采样"
[XTrain, YTrain] = makeWindowsTrain(trainSeqs(idxTr),  L, S, RULcap, true);
[XVal,   YVal  ] = makeWindowsTrain(trainSeqs(idxVal), L, S, RULcap, false);

% 测试：每台只取最后窗口的真值作对比；我们会在推理时用最后 K 个窗口做加权平均
[XTest,  YTest ] = lastWindowForTest(testSeqs, rulTest, L, RULcap);

fprintf('Train windows: %d, Val windows: %d, Test units: %d, Features: %d\n', ...
        numel(XTrain), numel(XVal), numel(XTest), F);

%% -------------------- 构建 SAE-TCN（含轻量注意力） --------------------
lgraph = buildSAETCN(F, D_embed, E1, E2, numBlocks, ksize, drop, L);
% figure; plot(lgraph); title('SAE-TCN with Lightweight Attention');

%% -------------------- 训练（带验证 & 早停） --------------------
options = trainingOptions('adam', ...
    'MaxEpochs',           maxEpochs, ...
    'MiniBatchSize',       miniBatchSize, ...
    'InitialLearnRate',    learnRate, ...
    'L2Regularization',    1e-4, ...
    'GradientThreshold',   1, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      {XVal, YVal}, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience',  5, ...        % 连续5次验证无提升即早停
    'Plots',               'training-progress', ...
    'Verbose',             true);

net = trainNetwork(XTrain, YTrain, lgraph, options);

%% -------------------- 测试评估（末端 K 加权平均） --------------------
Kavg  = 9;  % 使用最后5个窗口；可试 7 或 9（推理时间线性增加）
YPred = predictLastK(testSeqs, rulTest, net, L, RULcap, Kavg);

rmse  = sqrt(mean((YPred - YTest).^2));
mae   = mean(abs(YPred - YTest));
fprintf('FD001  RMSE = %.3f,  MAE = %.3f\n', rmse, mae);

% 散点对角图
figure; 
scatter(YTest, YPred, 25, 'filled'); grid on; hold on;
mmax = max([YTest; YPred])+1;
plot([0 mmax],[0 mmax],'k--','LineWidth',1);
xlabel('True RUL'); ylabel('Predicted RUL');
title(sprintf('SAE-TCN (Attn) on FD001 (RMSE=%.2f, MAE=%.2f)', rmse, mae));

%% -------------------- 单机 RUL 曲线可视化 --------------------
for i = 1:15
    plotRULCurveForUnit(testSeqs, rulTest, i, net, L, RULcap);
    saveas(gcf, sprintf('RUL_curve_unit%03d.png', i));
end

function lgraph = buildSAETCN(F, D, E1, E2, numBlocks, ksize, drop, L)
    % ----- SAE：逐时刻 1×1 卷积共享编码 -----
    layers = [
        sequenceInputLayer(F, 'Name','input', 'MinLength', L)
        convolution1dLayer(1, E1, 'Name','enc1', 'Padding','same')
        layerNormalizationLayer('Name','enc1_ln')
        reluLayer('Name','enc1_relu')
        dropoutLayer(drop,'Name','enc1_do')

        convolution1dLayer(1, E2, 'Name','enc2', 'Padding','same')
        layerNormalizationLayer('Name','enc2_ln')
        reluLayer('Name','enc2_relu')
        dropoutLayer(drop,'Name','enc2_do')

        convolution1dLayer(1, D,  'Name','enc3', 'Padding','same')
        layerNormalizationLayer('Name','enc3_ln')
        reluLayer('Name','enc3_relu')
    ];
    lgraph = layerGraph(layers);

    % ----- 轻量注意力：1×1 卷积 + Sigmoid 形成逐时刻、逐通道门控 -----
    attn = [
        convolution1dLayer(1, D, 'Name','attn_conv', 'Padding','same')  % 生成门控 logits
        sigmoidLayer('Name','attn_sig')
        multiplicationLayer(2, 'Name','attn_mul')                        % 与原特征逐元素相乘
    ];
    lgraph = addLayers(lgraph, attn);
    % 连接：enc3_relu → attn_conv → attn_sig → attn_mul(in2)
    lgraph = connectLayers(lgraph, 'enc3_relu', 'attn_conv');
    
    % 同时把原特征送到 attn_mul(in1)
    lgraph = connectLayers(lgraph, 'enc3_relu', 'attn_mul/in2');

    inName = 'attn_mul';   % 注意力输出作为 TCN 的输入
    dil = 1;

    % ----- TCN 残差堆叠（SAME padding + 膨胀卷积）-----
    for b = 1:numBlocks
        prefix = sprintf('tcn%d', b);
        blk = tcnBlock(prefix, D, ksize, dil, drop);
        lgraph = addLayers(lgraph, blk);
        lgraph = connectLayers(lgraph, inName, [prefix '_conv1']);
        lgraph = connectLayers(lgraph, inName, [prefix '_add/in2']); % 残差短接
        inName = [prefix '_out'];
        dil = dil * 2;  % 指数扩感受野
    end

    % ----- Head：GAP → FC → 标量回归 -----
    head = [
        globalAveragePooling1dLayer('Name','gap')  % 移除时间维，匹配标量标签
        fullyConnectedLayer(64, 'Name','fc1')
        reluLayer('Name','fc1_relu')
        dropoutLayer(drop,'Name','fc1_do')
        fullyConnectedLayer(1, 'Name','fc_out')
        regressionLayer('Name','regression')
    ];
    lgraph = addLayers(lgraph, head);
    lgraph = connectLayers(lgraph, inName, 'gap');
end
