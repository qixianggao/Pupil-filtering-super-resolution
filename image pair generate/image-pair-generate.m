clear all 
load('C:\Users\Administrator\Nutstore\1\我的坚果云\光瞳滤波\光瞳matlab\超分辨图像\球差\PSFw1f1.mat')
load('C:\Users\Administrator\Nutstore\1\我的坚果云\光瞳滤波\光瞳matlab\超分辨图像\球差\PSFw2f1.mat')
load('C:\Users\Administrator\Nutstore\1\我的坚果云\光瞳滤波\光瞳matlab\超分辨图像\球差\PSFw3f1.mat')
% load PSFw2f1.mat
% 
% load PSFw3f1.mat
% %球差 慧差 像散resize
% PSFw1f1 = imresize(PSFw1f1,[85,85]);
% 
% PSFw2f1 = imresize(PSFw2f1,[109,109]);
% 
% PSFw3f1 = imresize(PSFw3f1,[132,132]);


% %real resize
PSFw1f1 = imresize(PSFw1f1,[86,86]);

PSFw2f1 = imresize(PSFw2f1,[110,110]);

PSFw3f1 = imresize(PSFw3f1,[133,133]);
%pf resize
% PSFw1f1 = imresize(PSFw1f1,[96,96]);
% 
% PSFw2f1 = imresize(PSFw2f1,[123,123]);
% 
% PSFw3f1 = imresize(PSFw3f1,[150,150]);
imgPath ='D:\DIV2K\GT_220818\';

psf_Path = 'D:\DIV2K\real_valid\';
% resize_Path = 'D:\DIV2K\spherical\';
gt_Path='D:\DIV2K\GT_resize\';
imgDir = dir([imgPath '*.png']); % 遍历所有jpg格式文件
for i = 1:length(imgDir) % 遍历结构体就可以一一处理图片了
Uo = imread([imgPath imgDir(i).name]); %读取每张图片
Uo=imresize(Uo,[256,256]);
U2=im2uint8(Uo);
imwrite(U2,[gt_Path imgDir(i).name(1:end-3) 'png'])
Uo=double(U2);
B=double(Uo(:,:,1));
G=double(Uo(:,:,2));
R=double(Uo(:,:,3));
% B=imresize(B,[256,256]);
% R=imresize(R,[256,256]);
% G=imresize(G,[256,256]);
B1 = imfilter(R, PSFw1f1, 'conv', 'circular');
G1 = imfilter(G, PSFw2f1, 'conv', 'circular');
R1 = imfilter(B, PSFw3f1, 'conv', 'circular');
R1 = R1 - min(R1(:));
G1 = G1 - min(G1(:));
B1 = B1 - min(B1(:));
U=cat(3,R1/max(R1(:)),G1/max(G1(:)),B1/max(B1(:)));
U=im2uint8(U);
imwrite(U,[psf_Path imgDir(i).name(1:end-3) 'png'])
% U1 = imresize(U,0.5);
%     imwrite(U1,[resize_Path imgDir(i).name(1:end-3) 'png'])
end