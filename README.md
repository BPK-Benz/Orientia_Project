# Orientia_Project

  This is an analysis part of the HighThroughput of Orientia project (whole genome) to find the hit genes. Normally, hit selection is the direct comparison between the effect of the knockdown gene and that of the reference gene (scramble gene) and there is only an independent parameter considered for each biological process. Two favored techniques are 1) fold change for a replicate in an experiment and 2) strictly standardized mean difference (SSMD) for many replicates in an experiment. Therefore, SSMD is utilized in this experiment because there are three replicates.

  In this experiment, bacteria entry and bacteria translocations are two interesting main processes, and each process can be divided into two sub-processes: inhibit and enhance respectively. The independent variable for the entry and the translocation is the number of bacteria per infected cell (Bac/Inf) and the number of bacteria in the nucleus per infected cell (Nucbac/Inf) respectively.

  From investigating our results, it makes us would like to present an alternative way (regression methods) to find other hit genes that eliminate confounding effects before performing the conservative hit selection. Moreover, we prove the difference between hit genes from both analysis by studying biological pathways and protein complexes contained in OMICs datasets from Metascape software (https://metascape.org/). 

  The overview of the analysis includes three tasks:
1) Conservative analysis: SSMD
2) Adaptive analysis: Regression methods with SSMD
3) Combination hit genes from two analysis for the Bacteria entry and Bacteria translocation 
