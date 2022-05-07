# CS520-Final-Project
The final project of Umass CS520.

This project mainly studies a system that uses machine learning technology to predict the winning rate of League of Legends games. The system predicts the winning percentage of the game based on the lineup of both sides before the game starts as input. The models used in the system include binary logistic regression model and multilayer perceptron (neural network) model. The training and test data of the model are obtained by crawling the officially given API through web crawler technology. The model of the system is built with the sklearn library, and 240,000 games have been trained. Therefore, the system implements a system that can analyze the strength of the lineup selected by both sides in the game, so as to predict the winning rate of the game.

Now the System functional requirements analysis has been upload in folder Documents.

We trained three models (Logistic model, mlp model and SVM model respectively) and compared the prediction accuracy。

About Logistic model, 
    By calling the LogisticRegression method in the sklearn library, the basic model framework can be built, and before this, further processing of the input model data   is required. This step is called data preprocessing. Reference the StandardScaler method from sklearn.preprocessing to preprocess the data. Through this step, the raw   data has been processed into normalized data. The characteristic data variance of these normalized data is 1 and the mean is 0. The main purpose of data preprocessing   is to ensure that the predicted results will not be dominated by some extreme data, which will cause the predicted results to swing too much.
    After data preprocessing, call the LogristicRegression function to initialize the Logistic model as described above. There are many parameters that need to be tuned   in this step. And there are two parameters that I pay more attention to, one is maxiter and the other is C.
    Maxiter represents the maximum number of iterations of the model. Generally speaking, the higher the number of iterations, the more accurate the test results are,     but an excessively high number of iterations will cause over-fitting problems, which will reduce the accuracy during testing. The number of iterations of this model     is 200, so the value of the maxiter parameter is set to 200.
    C is the penalty coefficient in the Logistic model, the reciprocal of the regularization coefficient λ, and the value must be a positive number. The penalty           coefficient is also called the penalty term. The size of C marks the level of the regularization strength of the model, and its value is negatively related to the       regularization strength. Higher regularization strength will make the separation hyperplane more accurate, but it will also bring about overfitting problems.             Therefore, how to choose the penalty coefficient is an important issue in debugging the model. This step will be further explained later.
    Up to the previous step, the logistic model has been initialized.
    
About mlp model, 
    Like the construction of the Logistic model, the mlpc model needs to preprocess the data before it is built.
    The mlp model is initialized by the MLPClassifier method in sklearn.neural_network after preprocessing. Like the Logisitic model, the Mlp model also has a lot of       optional parameters in the initial trial. Here, the values of three parameters are mainly changed: hidden_layer_sizes; solver and max_iter.
    
About SVM model,
    The initialization and training of the support vector machine model are similar to the previous two models, which are done by calling the functions of the sklearn   library, so I will not repeat them.
    The main content is the initial test accuracy of the support vector machine model and the process of adjusting parameters.
    
