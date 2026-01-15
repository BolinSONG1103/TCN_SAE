function blk = tcnBlock(prefix, ch, k, d, drop)
    % 单个 TCN 残差块：Conv(dilated, same) → LN → ReLU → DO → Conv → LN → ReLU → DO → Add → ReLU
    blk = [
        convolution1dLayer(k, ch, 'DilationFactor', d, 'Padding','same', 'Name', [prefix '_conv1'])
        layerNormalizationLayer('Name', [prefix '_ln1'])
        reluLayer('Name', [prefix '_relu1'])
        dropoutLayer(drop, 'Name',[prefix '_do1'])

        convolution1dLayer(k, ch, 'DilationFactor', d, 'Padding','same', 'Name', [prefix '_conv2'])
        layerNormalizationLayer('Name', [prefix '_ln2'])
        reluLayer('Name', [prefix '_relu2'])
        dropoutLayer(drop, 'Name',[prefix '_do2'])

        additionLayer(2, 'Name', [prefix '_add'])
        reluLayer('Name', [prefix '_out'])
    ];
end