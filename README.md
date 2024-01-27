# **CREDIT CARD FRAUD TRANSACTION DETECTION**
---

## **INTRODUCTION**
**1.   Background**

> Credit card fraud is a significant concern for financial institutions and their customers. In the United States alone, credit card fraud losses were estimated to be $11 billion in 2020. Fraudulent transactions can be difficult to detect since they often involve small amounts and can be mixed with legitimate transactions. However, detecting fraud is crucial as it can help prevent financial losses and protect customers' sensitive information.



**2.   Motivation**


> The rise of online shopping and contactless payments has made credit card fraud detection more challenging. Fraudsters have become more sophisticated and can use stolen card information to make unauthorized purchases, leaving banks and their customers vulnerable. It is important that credit card companies can recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. Therefore, there is a need for advanced fraud detection systems that can quickly and accurately detect fraudulent transactions.



**3.   Goal**


> The goal of credit card transaction fraud detection is to develop a model that can accurately distinguish fraudulent transactions from legitimate ones. This can be achieved using various machine learning techniques such as Random Forest Algorithm, Multi-Layer Perceptron, Logistic Regression, K-Means Clustering, etc.


## **METHODOLOGY** 

> Credit card fraud detection necessitates the use of an ML-based algorithm to accomplish feature extraction and model evaluation. In this project, three machine learning techniques: Random Forest, Logistic Regression, and XGBoost are compared and analysed by utilizing the dataset from Kaggle that contains transactions made by credit cards September 2013 by European cardholders to create a fraud transaction detection system.


Following is the methodology employed by this project:

methodology.png

### **`1. EXPLORATORY DATA ANALYSIS`**

> Credit card transactions by European cardholders in September 2013 are contained in this dataset. A total of 492 frauds were detected out of 284,807 transactions that occurred over a two-day period. There is a high degree of unbalance in the dataset. There are 0.172% of all transactions that are classified as positive (frauds).

> Several numerical variables are contained in it, which are the results of a PCA transformation. Data features, background information, and original features have unfortunately been withheld due to confidentiality issues. With PCA, the principal components are V1, V2, ... V28. The table below outlines the features that were not calculated through the PCA algorithm.

| Sr no | Feature | Description |
|-------|---------|-------------|
| 1     | Time    | Records the duration between the first and last transaction in the dataset |
| 2     | Amount  | Amount of the transaction |
| 3     | Class   | Class 0: Non-Fraudulent Transactions <br> Class 1: Fraudulent Transactions |



> Several Python functions, namely head, describe, and info, were used to analyze the data's features and learn more about the structure and contents.

Description of Data.png

Initial Data.png

Class Distribution.png

Distribution of Column Amount.png

Distribution across V1.png

Correlation with Class column.png

Distribution of column 'Time'.png

Skewness Disribution.png


### **`2. DATA OBSERVATIONS`**

*   The data includes 8 Rows and 31 Columns.

*   The Columns Names from V1 to V28 are hidden as it contains people's confidential transactional information. Hence, they are already converted to PCA Vectors.
*   We have Time and Amount column, and the Class column is the target column and represents if the transaction is a fraud or not. '0' means it's not a fraud and '1' means it's a fraud. On further analysis, we can see that the number of frauds (492) in the data are much less than the legit transactions (284315). This indicates the data is highly unbalanced and might require under sampling or oversampling.
*   The data does not have any null values in any of the columns.
*   From the histograms, we can observe that the data has maximum transactional amounts below 100 with a mean of 88. The values of PCA vector V1 are maximum between -5 to 5.
*   Further, I have also checked the skewness across each column by plotting its distribution across histogram and have fixed it using a Power Transformer 





### **`3. FEATURE ENGINEERING`**



*   Feature engineering is a crucial step in the data preprocessing pipeline, as it involves creating new features or transforming existing ones to capture relevant information and patterns within the data. 
*   I introduced a new feature named 'Amount_Relative' to better represent the transaction amount's significance relative to the class of transactions (fraudulent or legitimate). 
*   'Amount_Relative' was created by calculating the ratio of each transaction's amount to the average transaction amount within its respective class. 
*   This approach allows the models to consider the relative importance of the transaction amount, which can be particularly informative in distinguishing between different classes of transactions.



### **`4. DATA PRE-PROCESSING`**

Overcoming the problem of an imbalanced dataset by using sampling techniques

>  The data I am using is rather uneven, and I have found that there is a large difference between the number of real and fraudulent transactions. To solve this problem, I chose to produce a more balanced dataset by oversampling the fraudulent values and under sampling the legitimate values.

>  I created a new dataset named "test_over" by combining the original dataset of valid transactions with 3000 instances randomly chosen from the original dataset of fraud transactions using replacement. This strategy aids in improving the dataset's representation of the minority class (fraud).

>  Then, to choose examples from the majority class (legal transactions), which is closest to the minority class (fraud transactions), in terms of distance, I applied the NearMiss method with 10 nearest neighbours. 

>  Then, using a bar chart with the x-axis representing the class labels (legit and fraud) and the y-axis representing the amount of instances, I printed the number of legitimate and fraudulent values in the sampled dataset. 

>  After the oversampling and under sampling processes, this bar chart gives a clear visual representation of the achieved balance between the two classes.

Bar plot after sampling.png

### **`5. SPLITTING DATA INTO TRAINING AND TESTING`**

> Now that the sampling is completed and the data is now balanced, I will continue with splitting the data into Training and Testing. 70% of the total data will be used to train the ML model, with the remaining 30% used to test the model's accuracy.


### **`6. TRAINING AND TESTING MACHINE LEARNING MODELS`**

> I used three different machine learning algorithms and one deep learning algorithm. Trained and tested them all with the same dataset. These algorithms are as follows:



*   Random Forest Modeling: 
*   Logistic Regression Modeling:
*   eXtreme Gradient Boosting (XGBoost):
*   Multi-Layer Perceptron


## **RESULT AND ANALYSIS**

#### **Performance Evaluation**

For each model, I have:
1.	Compared the training set results v/s the testing set results
2.	Plotted Confusion Matrix and ROC Curve to determine outcomes

#### **`1.   Random Forest Modeling`**
#### **`2.   Logistic Regression Modeling`**
#### **`3.   XGBoost`**
#### **`4.   Multi-Layer Perceptron`**

## **CONCLUSION**

> In order to address the problem of data imbalance in the credit card fraud detection dataset, this research used three different models: Random Forest Classification, Logistic Regression, and XGBoost. The real values were under sampled to 3000 while the fraudulent values were oversampled to 3000 in order to balance the imbalance. Then, a 70:30 split between the training and test sets was applied to the dataset.

> The models were trained on the training set, and the accuracy of their predictions on the testing set was used to gauge their performance. When compared to the Logistic Regression model, which performed well in terms of accuracy, precision, and how it appeared on the Confusion Matrix and ROC Curves, the Random Forest Classifier and XGBoost models showed overfitting.

> Based on these results, it can be said that the Logistic Regression model performed better than the other two models and is the best choice for this project's evaluation of credit card fraud transactions.


## **FUTURE SCOPE**

> Since two of the models overfitted the data, we can explore more models to detect fraudulent credit card transactions.


## **DATASET LINK**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## **REFERENCES**

[1] Muaz, A., Jayabalan, M., & Thiruchelvam, V. (2020). A Comparison of Data Sampling Techniques for Credit Card Fraud Detection. International Journal of Advanced Computer Science and Applications, 11(6). https://doi.org/10.14569/ijacsa.2020.0110660

[2] Jake VanderPlas, Python Data Science Handbook.

[3] Shen, A., Tong, R., & Deng, Y. (2007). Application of Classification Models on Credit Card Fraud Detection. 2007 International Conference on Service Systems and Service Management. 

[4] GPT-3.5, codellama
