The wine data are the results of a chemical analysis of wines grown in the same region in Italy
but derived from three different cultivars. The analysis determined the quantities of 13 constituents 
found in each of the three types of wines. 

To evaluate the effectiveness of my model, I compared the baseline prediction model 
which just randomly outputs one type of wines given attributes. 
For this project, I used 5-fold cross-validation. 
5-fold Cross-validation is randomly divide the traning data D into 5 groups with the same size say {Di} i = 1.
During each iteration i for i ∈ {1, 2, 3, 4, 5}, select D −Di and Di as training data and testing data independently.
Finally, output the averaged testing accuracy as the output accuracy for a specific predicting model. 

When I do not use the Kfold cross validation the accuracy from the MLP is 56.79
But when I used kfold cross validation it comes with different out. 

