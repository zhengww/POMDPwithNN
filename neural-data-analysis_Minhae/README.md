# neural_data_analysis
This code is for neural data analysis using regression methods. 
All data files are located in `data` folder.


There are five core codes to perform neural data analysis
- `data_preprocessing.py`: this code transform `.pkl` file to `Pandas DataFrame` `.csv` file. 


In the following codes, I split the data set into `train_data`, `validation_data`, `test_data`. 
`test_data` is never touched during the training. I use [Stratified K-fold cross validation](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f) in order to avoid overfitting. 
During K-fold validation, `train_data` and `validation_data` are determined and used.

- `encoding.ipynb`: Estimate belief from neural signal
    - input: `r_df.csv` (neural signal from 300 neurons)  
    - output: `nb_df.csv` (estimated neural belief)
    - method: linear regression
    - WORKS VERY GOOD!
 
- `recoding`: Estimate the next belief from current belief and observations
    - input:
        - if you use POMDP data (for now): `recoding_pomdp_all_prev_df.csv` and `recoding_pomdp_all_now_df.csv`
        - ideally with neural data: `recoding_neural_all_prev_df.csv` and `recoding_neural_all_now_df.csv`
    - output: `recoding_belief_results_df.csv` (estimated future belief)
    - method: Autoregression - this is linear regression between two time steps. 
    - there are two versions in codes:
        -`recoding_wo_RBF.ipynb`: no RBF is used. 
        - `recoding_KRR.ipynb`: RBF is used using sklearn built-in function: [Kernel Ridge Regression (kernel ='rbf')](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html). 
        (Warning: this code runs pretty slow compared to others. So if you need to handle big data size, plan in advance! 
        or decrease # of folds (K) in K-fold validation)
        - `recoding_manualRBF.ipynb`: RBF is manually coded by me. So we can customize center locations for nonlinear transform. 
        Everything is the same as `recoding_wo_RBF.ipynb` but RBF.
        - `recoding_KRR.ipynb` works the best, and pretty good!
        
- `decoding`: Find policy that returns action from neural belief and location.
    - input:
        - if you use POMDP data (for now): `pomdp_decoding_data.csv` which includes belief and location
        - ideally, `nb_df.csv` (estimated neural belief - obtained from `encoding_v3.ipynb`) and 
        `neural_decoding_data` which includes behavior belief and location.    
    - output: `decoding_a_results_df.csv` (estimated action)
    - method: multinomial logistic regression
    - there are three versions in codes 
        - `decoding_woRBF.ipynb`: no RBF is used. Iris data is used to test the code
        - `decoding_KRR.ipynb`: RBF is used using sklearn built-in function: Kernel Ridge Regression (kernel ='rbf'). 
        This is linear regression not logistic regression Since sklearn does have multinomial logistic regression with kernel. 
        - `decoding_manualRBF.ipynb`: RBF is manually coded by me. So we can customize center locations for nonlinear transform. 
        multinomial logistic regression is used the same as `decoding_woRBF.ipynb`. <br />
    - `decoding_KRR.ipynb` performs the best, but not good enough (I guess..)
    - I suspect that the reason why the logistic regression in decoding doesn't work well is imbalanced data. 
    If you see our actions, dominant of actions is ZERO. This causes the regression model prefer to choose ZERO rather than other actions.
    Please see this reference [Link](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) 

Additional code: `belief-action-relationship.ipynb`: you can play around with your data to test whether it is reasonable or trash.

The results so far I have gotten: 08/11/2019


![Encoding: BOX1](./figures/encoding_box1.jpg) 
![Encoding: BOX2](./figures/encoding_box2.jpg) 
![Recoding: BOX1](./figures/recoding_KRR_box1.jpg) 
![Recoding: BOX2](./figures/recoding_KRR_box2.jpg) 
![Recoding_DELTA_BOX1](./figures/recoding_DELTA_KRR_box1.jpg)
![Recoding_DELTA_BOX2](./figures/recoding_DELTA_KRR_box2.jpg)
![Decoding](./figures/decoding_KRR.jpg) 

