# Bi-Level-Thresholding
## Introduction

 Bi-level thresholding of images occupies a very important field in digital image processing. It is used extensively in daily life. 
 If the given image does not have a bimodal intensity histogram, it will cause segmenting mistake easily for trivial bi-level algorithms. 
 In order to solve this problem, a new algorithm is proposed in the paper "Global Automatic Thresholding with Edge Information and Moving Average on Histogram" 
 by Yu-Kumg Chen and Yi-Fan Chang. The proposed algorithm uses the theory of moving average on the histogram of the fuzzy image, 
 and then derives the better histogram. Since use only one thresholding value cannot solve this problem completely, 
 the edge information and the window processing are introduced in this paper for advanced thresholding. 
 Thus, a more refine bi-level image is derived for corneal topography images. 
 
 
 ## Technologies
 - OpenCV
 - Matplotlib.pyplot
 - numpy
 
 ## Fucntion
 biLevelThresholding(image , m , th)
 
 ### Input
 - image:- image in array form
 - m:- window size for calculating moving avg.
 - th:- threshold value for sobel output discrimination. If value is great than "th", s(x, y) is set to 1 else 0
 
 ### Output
 - bi_img:- Bi-level image
 - threshold:- Otsu's threshold value
 
 ## Result
 
![input_img.png](attachment:input_img.png)
 
![Output_img.png](attachment:Output_img.png)
