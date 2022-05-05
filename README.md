# Orientia_Project

This is a part of HighThroughput of Orientia project (whole genome). 
The highlight is presenting regression analysis with machine learning to predict a key parameter of Biology mechanism under the same environment of scramble gene before performing a tradition evalustion e.g. strictly standardized mean difference (SSMD technique) to find hit genes. 


There are four main steps for searching hit genes of both Bacteria entry mechanism and Bacteria Translocation mechanism including:
1. Cleaning all of genes data
2. Conservative analysis to find hit gene by only using strictly standardized mean difference (SSMD technique). This is a conservative technique that compare only a key parmeter such as number of bacteria per cell.
3. Adaptive analysis to find hit gene by using both regression analysis and ssmd technique. In here, the comparison should be the same environment that is the key concerned aspect before using the tradition comparison.
  
	3.1 There are 8 parent formulars: Multiple_Linear_Regression, Polynomial, Exponential_and_Reciprocal, Exponential2_and_Polynomial, Exponential_and_Linear, Exponential_and_Polynomial2_Degree21, Polynomial_Degree2, Logistic
  
	3.2 10 Fold cross validataion is used for find the best optimal parameters of each parent function by minimum L2-Norm and for find the best optimal equation by maximum R-square
  
	3.3 The best parent function is used for predicting a key predicted parameter
  
	3.4 SSMD Comparison between scramble gene (reference gene) and knocked down gene are evaluated. These are also divided into two sub-process: inhibit and enhance
	
4. Combination of each mechanism: Finally, we got hit genes for Bacteria entry mechanism (inhibit and enhance sub-process) and hit genes for Bacteria Translocation mechanism (inhibit and enhance sub-process)
