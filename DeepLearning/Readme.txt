The data set I worked with a CIFAR10, it consists of 50,000 32*32 color training images, 
with 10 different categories. The testing set has 10,000 color images. 
I used the Python Deep Learning Packages to help me to build up a model for multi-class classification.

Deep Neural Network 
For this project, I build up deep neural networks with different typer-parameters to solve 
the 10 - class classification problem. I made one baseline DNN model and include two other DNN models 
with better performance that is fine tuned. 


From this project I used DNN model to sorting the pictures from cifar10. For my base model, I just used Dense to sorting,
and for the first implementation I added Conv2D to raise up my accuracy, for the second implementation I added Conv2D and Maxpooling 
to raise my accuracy. When I used Base model, the accuracy is above 50%, and when we use the conv2D for my first implementation model, 
the accuracy was about 80% which was the highest. However, from the second implemantation, when I used Maxpooling with 
Conv2D accuracy was decreaseed but the time complexity was increased. 