
clear all
close all
clc

%Images are saparated by Train and Test Folder in Example1 folder
imds = imageDatastore('/Users/rinrada.w/Downloads/Demo2/Facial eiei/' , ...
     'IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

net = resnet50;
net.Layers

inputSize = net.Layers(1).InputSize

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'loss3-classifier';  %see detail in net.Layers command that what is the layer of 'fc7'
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

accuracy = mean(YPred == YTest)
plotconfusion(YTest,YPred)

save FeatureAlexTrain.mat featuresTrain     
save FeatureAlexTest.mat featuresTest
