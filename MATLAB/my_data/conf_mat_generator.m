%% Confusion Matrix:
% matrix = [153,0,0,0,0;1,105,0,0,0;0,3,102,0,0;0,0,0,105,0;0,0,0,2,103];
M =  [[121   5   2   1   0]
 [ 10  90   7   2   5]
 [  1   9 102   1   0]
 [  0   1   7  97   8]
 [  0   7   3   6  97]];
C3(1,:,:) = M;
% ATN = sum(squeeze(sum(C3,1)),2);%sum(sum(C));
ATN = sum(matrix, 2);
SumDim1 = squeeze(sum(C3,1));
figure;
PercentC = zeros(5,5);
for i=1:5
    PercentC(i,:) = (SumDim1(i,:)./(ATN(i)))*100;
end
labels = {'Alpha','11.1','12.5','15.2','16.7'};
heatmap(PercentC, labels, labels, '%0.2f%%','Colormap','jet','ShowAllTicks',0,'UseLogColorMap',true,'Colorbar',true,'ColorLevels',30,'MaxColorValue',100,'MinColorValue',0);
figure; 
heatmap(PercentC, labels, labels, [],'Colormap','jet','ShowAllTicks',0,'UseLogColorMap',true,'Colorbar',true,'ColorLevels',30,'MaxColorValue',100,'MinColorValue',0);
OverallAccuracy = sum(diag(PercentC))/5
xlabel('Predicted'),ylabel('Actual');
title(['Confusion Matrix for conv-PSDA method, Accuracy = ', num2str(OverallAccuracy), '%']); 