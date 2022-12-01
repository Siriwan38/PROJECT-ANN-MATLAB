clear all
close all
clc
%Copy Images to Folder Example2 by do not saparate train and test
imds = imageDatastore('/Users/siriwanpratan/Downloads/MATLAB_PROJECT', ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

    
%Divide the data into training and validation data sets. 
%Use 70% of the images for training and 30% for validation. 
%splitEachLabel splits the images datastore into two new datastores.   
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%Randomly show some images
% numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end

net = alexnet;

analyzeNetwork(net)

%The first layer, the image input layer, requires input images of 
%size 227-by-227-by-3, where 3 is the number of color channels.
inputSize = net.Layers(1).InputSize

%Replace Final Layers

%The last three layers of the pretrained network net are configured 
%for 1000 classes. These three layers must be fine-tuned for the 
%new classification problem. Extract all layers, except the last three, 
%from the pretrained network.

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2)
    softmaxLayer
    classificationLayer];

%Train the Network

% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',50, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',0.01, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',500, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

plotconfusion(YValidation,YPred)
