% Copyright (c) 2014, Sanyam Garg. All rights reserved.
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
% * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% The given code finds the HOG feature vector for any given image. HOG
% feature vector/descriptor can then be used for detection of any
% particular object. The Matlab code provides the exact implementation of
% the formation of HOG feature vector as detailed in the paper "Pedestrian detection using HOG" by Dalal and Triggs.
% INPUT => im (input image)
% OUTPUT => HOG feature vector for that particular image
% Example: Running the code
% >>> im = imread('cameraman.tif');
% >>> hog = hog_feature_vector (im);

function [feature] = HOG(im)
if size(im,3)==3,im=rgb2gray(im);end
im=double(im); rows=size(im,1); cols=size(im,2);
Ix=im; Iy=im; %Basic Matrix assignment
% Gradients in X and Y direction. Iy is the gradient in X direction and Iy is the gradient in Y direction
for i=1:rows-2, Iy(i,:)=(im(i,:)-im(i+2,:));end
for i=1:cols-2, Ix(:,i)=(im(:,i)-im(:,i+2));end

%gauss=fspecial('gaussian',8); %% Initialized a gaussian filter with sigma=0.5 * block width.    
angle=atand(Ix./Iy); % Matrix containing the angles of each edge gradient
angle=imadd(angle,90); %Angles in range (0,180)
magnitude=sqrt(Ix.^2 + Iy.^2);
% figure(2); subplot(1,2,1);imshow(uint8(angle));title('angle');
% subplot(1,2,2);imshow(uint8(magnitude));title('magnitude');

% Remove redundant pixels in the image. 
angle(isnan(angle)) = 0;
magnitude(isnan(magnitude)) = 0;

feature=[]; %initialized the feature vector
% Iterations for Blocks
for i = 0: rows/8 - 2,
    for j = 0: cols/8 - 2,        
        mag_patch = magnitude(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);
        %mag_patch = imfilter(mag_patch,gauss);
        ang_patch = angle(8*i+1 : 8*i+16 , 8*j+1 : 8*j+16);        
        block_feature=[];
        
        %Iterations for cells in a block
        for x = 0:1
            for y = 0:1
                angleA = ang_patch(8*x+1:8*x+8, 8*y+1:8*y+8);
                magA   = mag_patch(8*x+1:8*x+8, 8*y+1:8*y+8); 
                histr  = zeros(1,9);
                
                %Iterations for pixels in one cell
                for p=1:8,
                    for q=1:8,
                        alpha = angleA(p,q);                        
                        % Binning Process (Bi-Linear Interpolation)
                        if alpha>10 && alpha<=30
                            histr(1)=histr(1)+ magA(p,q)*(30-alpha)/20;
                            histr(2)=histr(2)+ magA(p,q)*(alpha-10)/20;
                        elseif alpha>30 && alpha<=50
                            histr(2)=histr(2)+ magA(p,q)*(50-alpha)/20;                 
                            histr(3)=histr(3)+ magA(p,q)*(alpha-30)/20;
                        elseif alpha>50 && alpha<=70
                            histr(3)=histr(3)+ magA(p,q)*(70-alpha)/20;
                            histr(4)=histr(4)+ magA(p,q)*(alpha-50)/20;
                        elseif alpha>70 && alpha<=90
                            histr(4)=histr(4)+ magA(p,q)*(90-alpha)/20;
                            histr(5)=histr(5)+ magA(p,q)*(alpha-70)/20;
                        elseif alpha>90 && alpha<=110
                            histr(5)=histr(5)+ magA(p,q)*(110-alpha)/20;
                            histr(6)=histr(6)+ magA(p,q)*(alpha-90)/20;
                        elseif alpha>110 && alpha<=130
                            histr(6)=histr(6)+ magA(p,q)*(130-alpha)/20;
                            histr(7)=histr(7)+ magA(p,q)*(alpha-110)/20;
                        elseif alpha>130 && alpha<=150
                            histr(7)=histr(7)+ magA(p,q)*(150-alpha)/20;
                            histr(8)=histr(8)+ magA(p,q)*(alpha-130)/20;
                        elseif alpha>150 && alpha<=170
                            histr(8)=histr(8)+ magA(p,q)*(170-alpha)/20;
                            histr(9)=histr(9)+ magA(p,q)*(alpha-150)/20;
                        elseif alpha>=0 && alpha<=10
                            histr(1)=histr(1)+ magA(p,q)*(alpha+10)/20;
                            histr(9)=histr(9)+ magA(p,q)*(10-alpha)/20;
                        elseif alpha>170 && alpha<=180
                            histr(9)=histr(9)+ magA(p,q)*(190-alpha)/20;
                            histr(1)=histr(1)+ magA(p,q)*(alpha-170)/20;
                        end                                        
                    end
                end
                block_feature=[block_feature histr]; % Concatenation of Four histograms to form one block feature                                
            end
        end
        % Normalize the values in the block using L1-Norm
        block_feature=block_feature/sqrt(norm(block_feature)^2+.01);
        feature=[feature block_feature]; %Features concatenation
    end
end

feature(isnan(feature))=0; %Removing Infinitiy values

% Normalization of the feature vector using L2-Norm
feature=feature/sqrt(norm(feature)^2+.001);
for z=1:length(feature),if feature(z)>0.2, feature(z)=0.2;end;end
feature = feature/sqrt(norm(feature)^2+.001);              
