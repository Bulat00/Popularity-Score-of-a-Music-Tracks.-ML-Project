# Popularity score of music tracks
- Michele Tresca 
- Greta Kichelmacher 
- Silvia Conti

# Introduction 
Due to the rising popularity of digital music platforms, understanding the characteristics that contribute to a song's popularity has become increasingly important, not only for music producers and artists but also for the platforms themselves. Thanks to the  dataset provided to us, we were able to perform an in-depth analysis of the factors influencing a music track's popularity. Based on our findings, we aim to offer valuable recommendations for music producers and artists.


#  EDA
Firstly, we import a selection of libraries that will facilitate our data analysis:

- *"sklearn"*: This library offers a broad array of machine learning algorithms, tools for data preprocessing, and functions for performance evaluation.
- *"numpy"*: It's a powerful tool for scientific computation in Python.
- *"pandas"*: This library provides easy-to-use data structures and data analysis tools.
- *"matplotlib"* and *seaborn*: Both libraries are used for data visualization, with Seaborn being built on top of Matplotlib and offering a higher level of abstraction for statistical plots.

Next, we import our CSV file and read the data into a DataFrame. 
Using the `*df.head()`* function, we can quickly display the initial rows of our DataFrame, giving us a glimpse into our dataset. 

To better understand the structure and type of our data, we use the `df.info()` function. This helps us identify the data types for each of the columns in our DataFrame. Here's what we find:

- The columns Unnamed, popularity, duration_ms, key, time_signature, and mode are of integer type.
- `track_id`, `artists`, `album_name`, `track_name`, and `track_genre` that are objects, specifically, they are strings.
- `explicit` is a boolean value.
- `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `livness`, `valence,` and `tempo` are float values.

This breakdown gives us a clear understanding of our dataset's structure and the types of data it contains.
Using the `df.shape` function, we ascertain the dimensions of our DataFrame, which contains 114,000 rows and 21 columns.

Based on the feature descriptions, it initially appeared that `track_id` would serve as a unique identifier for each track. However, upon examining the unique values of `track_id`, we discovered that this was not the best choice. Upon closer inspection, we observed that rows with the same `track_id` could have different values for `track_genre` and `popularity`. This led us to the realization that the popularity of a song can vary depending on its genre.

Next, we employ the `df.describe()` function to generate summary statistics for our data, which includes mean, standard deviation (std), minimum (min), and maximum (max) values. From this summary, we observe that the *mean popularity is* *33%*, with a *standard deviation of 22%* (The minimum popularity is 0 and the maximum is 100) . This distribution suggests a leftward skew in the data, indicating that the majority of elements have low popularity, with only a few outliers enjoying very high popularity. Specific attention must be paid on `duration_ms`, that has the maximum value that is very far from the third quantile (the 75% of values reach at least 2.5 millions, while the range is 0 to 5 millions), and this could generate a lot of noise within the model. Furthermore by looking at the general difference between minimum and maximum values of different features, we can already say that we will have to perform some scaling in order to have a better performance.

We proceed to check for invalid values and discover *NaN* (Not a Number) values in the columns: `artists, album_name, and track_name`. 
Since these are string-type columns, we don't opt to replace these missing values with mean or median values as it wouldn't be meaningful. Considering that these NaN values represent only 3 rows out of the total 114,000, we decide to remove these rows. 
The loss of data is inconsequential in this context. By specifying *`inplace=True,`* we ensure that our function directly modifies the original DataFrame, effectively removing rows with missing values and overwriting the existing DataFrame.

Following this, we remove the first column labeled 'Unnamed' as it merely duplicates the index column, offering no new information.

Lastly, we checked for and eliminated any duplicates within our DataFrame in order to ensure accurate, consistent, and high-quality data for analysis and modelling. 
By using the `df.duplicated().sum()` function, we find that there are 450 duplicate entries. We proceed to remove these using the df.drop_duplicates *`(inplace=True)`* function, ensuring our data is free from redundancies.

We proceed to examine the correlation between variables by creating a correlation matrix. To make these correlations visually intuitive, we plot a heatmap which uses a color gradient to highlight the strength of correlations. Notably, the variables `valence`and `danceability` exhibit a strong correlation, as do `'loudness'` and `'energy'.` Interestingly, we observe that popularity does not have a direct or linear correlation with any other parameters.

By employing the `sns.pairplot(df)` function, we generate a scatter plot matrix, which visually represents the relationships between all pairs of variables in the DataFrame. This further confirms the absence of linear correlations. 

Subsequently, we plot the distributions of each parameter to gain a deeper understanding of our data. For instance, we find a significant number of zeroes in the `popularity` distribution. From the 'duration_ms' histogram, we observe that while the values range from 0 to 5 millions, they are primarily concentrated between 0 and 1. As for the `mode` histogram, since it's a boolean variable, the values are either 0 or 1. Furthermore we can see that many histograms are skewed, and this is a further confermation that transformations need to be applied.

Next, we create some box plots, that are very useful to have a concise graphical representation of the key characteristics of feature distributions. We create some boxplots for selected variables termed `audio_features`, which share a common range of values between 0 and 1, and are hence comparable. 
From this plot, it's evident that the variables `danceability`, `energy`, `acousticness`, and `valence` demonstrate reasonable values, while other variables exhibit a significant number of outliers. Particularly, we notice a high concentration of data at lower values, with outliers at higher values. 
Then, we generate individual box plots for `Tempo`, `Loudness` and `duration_ms`. 
For the boxplot of `duration_ms`, we can easily see that the interquantile difference is very small, ranging from 0 to around 0.5, while few values are very far from those. 

By having in mind the previous discover that popularity for the same song (same `track_id`) can vary according to musical genre, we delve deeper into the top 10 music genres in terms of their average popularity. By calculating the average popularity score for each music genre, we group the DataFrame rows by genre. We then compute the mean and standard deviation of popularity for each genre and plot a histogram that associates the top 10 music genres with their popularity. 
This histogram reveals that the music genre with the highest average popularity (pop-film) has a lower standard deviation in comparison to another music genre (pop), which, despite its lower average popularity, has a higher standard deviation. We repeat this process for the top 10 artists.


# Data Preprocessing
Having previously noted that identical `track_id` can have different popularity levels and different genres, we might consider dropping duplicate track_id to eliminate some noise by associating each unique value with the highest popularity level for that `track_id`. We thought that if we only  kept the most popular version of each song, we could learn more about popular songs in order to recommend them. 
But we also thought  it  would  be  helpful to know  how  different  characteristics of music  affect  how  well a song  does, more specifically t might be more interesting to keep the same song in different genres and observe the variability of popularity levels. This could provide useful information on the relevance of genre in determining a song's popularity.
By having this in mind, we provide the code for data preprocessing that can be changed between True or False depending on the idea for the analysis. 
Moving forward the `explicit` column is converted to int64 to ensure compatibility with machine learning algorithms that require numerical inputs.
Then we've created a new DataFrame called `X` that includes all the columns from `df`, unless `popularity, track_id, track_name`. 
This DataFrame will serve as the input for training the machine learning model.  Lastly  `y` is created, containing only the `popularity` column from `df`, representing the target variable that the model aims to predict. 

### Data Binning and training and test split
As the distribution of the popularity score was not constant among different values we implemented a data binning methodology to partition the original dataset into a training and test set containing the same proportion of popularity Score values. We started by creating bins and labels to categorize the values in the `y` column into distinct categories. Then, a new column 'bins' is added to the DataFrame `X`, containing the assigned category for each value in `y`. Then we've splitted the data into training and testing sets, maintaining the class distribution based on the `bins` column. 
The features and target variable data are assigned to separate variables for both the *training and testing sets*. Then we've removed the `bins` from both the training and testing DataFrames. 

Moving forward we have transformed the `duration_ms` variable. More specifically we saw that the there were only 1136 values that exceed the 99th percentile (while there were too many before this threshold) and those are just increasing the noise in the model, therefore for any 'duration_ms' values that surpass the threshold are replaced with the threshold value. This approach ensures that excessively high 'duration_ms' values in the training data do not have a disproportionate impact on the model's learning process.


### Encoding
considering that the variable:

- `key` can assume only integer values ranges from 0 and 11
- `time_signature` only integer values ranges from 0 to 5 

We have decided to treat these two features as categorical variables based on their meaning and therefore we encoded them with the One Hot Encoding. After the OHE, we used the Leave One Out Encoder for `artists`, `track_genre` and `album_name`. That is because considering the general importance of the artist, the genre of the track and the name of the album, including this features inside our model will be very important.
After the encoding we have performed a code to check the correlation with the features that are not anymore categorical. According to the analysis, we observe that the variables `track_name`, `album_name` and `track_genre` exhibit significant correlation with the target variable. Additionally, there is a strong correlation between `album_name` and popularity. However, to retain the meaningful information for distinguishing popularity levels based on `'album_name'`, it is advisable not to drop this variable .

### Feature Scaling 
As first thing, we made a code that separates the columns containing dummy variables from the rest of the encoded data to avoid their scaling. This ensures that the scaling process does not affect the interpretability or the meaning of the dummy variables, which represent categorical information in the dataset. These columns are extracted into new DataFrames (`X_train_cat` and `X_test_cat`), while the original DataFrames (`X_train_encoded` and `X_test_encoded`) have these columns removed.
Then, we performed the feature scaling in order to standardize or normalize the numerical features in the dataset to ensure that they are on a similar scale. By using feature scaling, specifically the *StandardScaler,* we have achieved some benefits like:

- Normalization of the distribution
- Handling the outliers
- Mantaining the interpretability of the data
- Ensuring equal importance for all features

Lastly, After performing the standard scaler, we've made a code that creates sequential identifier columns (`merge` and `merge1`) for the rows in the scaled training and testing DataFrames.  These columns are then used to merge the categorical and numerical feature DataFrames with the scaled DataFrames, respectively. 
The resulting merged DataFrames contain both the categorical and numerical features, while the identifier columns are subsequently dropped.

# Clustering
To perform the task related to cluster analysis we have started by creating a new dataframe called `df_cluster`, which is an exact copy of the `X_train_scaled` dataframe. Then, we have setted  the number of unique music genres as the number of clusters, aiming to see if K-Means could identify groups of songs that correspond to music genres. Lastly, we created a new column called `cluster`, where we inserted the cluster membership of each song into the dataframe

Then we've performed the accuracy score.
Our analysis revealed that *the model's accuracy is very low (0.0001425438596491228),* indicating that the clusters identified by KMeans do not closely correspond to the original music genre labels. This *is not surprising*, given that KMeans is an unsupervised clustering algorithm and does not use labels during training. Additionally, music genres can be quite subjective and might not directly correspond to measurable audio features of a track.

However, we also computed the *Normalized Mutual Information (NMI)* between the clusters and the original labels, *obtaining a score of 0.4019656340208907*. NMI is a measure that quantifies the association between two categorical variables, and can be a better option than accuracy to evaluate clustering results. The relatively high NMI score suggests that there is some association between the clusters and the music genres, although it's not perfect.

In conclusion, while using KMeans to infer a track's genre based solely on its audio features can yield some interesting results, it doesn't appear to be a highly accurate technique. This could be due to the subjective nature of music genres, which might not be fully captured by audio features."

# Models
Before starting to perform the models we've imported the GridSearchCV and the performance metrics that we are going to use. 
For every model, the code will perform model evaluation by calculating R^2 and RMSE values for both the training and test sets in order to better assess the presence of over/underfitting. 

### Linear Regression
Before calculating the performance metrics on data, we had to create the linear regression model and to fit it on X_train_scaled, y_train. These metrics provide insights into the predictive performance of the model, indicating the proportion of variance explained by the model and the average prediction error. Those are our results:

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| *Training Set* | 0.7873274350786287 | 10.275844108326021 |
| *Test Set*     | 0.7152642774777251 | 11.893668564220878 |

Even if the relationship between the independent features and popularity was not linear, we can see that the results are not bad, and this happens due to the fact that we added some linearity with the categorical variables encoded with LeaveOneOut that we were not able to see previously.
Now we are going to perform some scatter plot analysis for `track_genre, artists` and ` album_name.`
The three scatter plots allow for exploring the relationship between `'y_train2'` and specific predictor variables, enabling the identification of potential correlations or patterns that can be useful for subsequent analysis and modeling. From this plots we can see that the relationship between album_name and popularity, and between artists and popularity is a little bit more linear. 

### Lasso Regression
Lasso Regression is a type of linear regression model that incorporates a regularization term to prevent overfitting, making it ideal for dealing with high-dimensional datasets. This method also performs feature selection by shrinking the coefficients of less important features to zero, and this could be very useful considering the general poorly correlation between independent features and the target. To optimize the model's performance, a GridSearchCV is used to adjust the alpha hyperparameter, balancing the model's fit to the training data and the reduction of model complexity via coefficient shrinkage.

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| *Training Set* | 0.7868394239628225 |  3.8877262943718374 |
| *Test Set*     |  0.713398294731186 | 9.604719179276078 |


However, as we can see, the results are very similar to the linear regression, and this suggest that the added regularization and feature selection capabilities of lasso regression were not beneficial for the specific dataset and modeling task at hand.

## Ensamble Methods

After our first two interpretable model, to improve our results we pass on the ensemble methods

### Gradient Boosting
The Gradient Boosting Regressor is a powerful algorithm that works in an iterative manner. Initially, it uses the entire training set, and with each subsequent iteration, it focuses more on the examples that were mispredicted in the previous one. 
To perform the model we have defined some parameters and then a GridSearchCV was made. After this we trained the model with training set and make prediction on both training and test sets.


|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| *Training Set* | 0.9695583759259719 |  3.8877262943718374 |
| *Test Set*     |  0.8143137970272326 | 9.604719179276078 |

After this, we plotted the residuals against the predicted values to see if there is overfitting in the datasets.
In the graph, the axes represent the predicted values and the residuals, respectively. 
The points on the graph represent the discrepancies between the predicted values and the actual values. The x-axis shows the predicted values, while the y-axis represents the residuals. As we can see from the large difference between the performance on the training and on the test and from the graph, our model is doing overfitting, therefore we had to simplify the model.
By considering that the best parameters of the GridSearchCV are {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300} in order to reduce the complexity we tried to reduce the max_depth and n_estimators tring to reach a less overfitted model.

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| *Training Set* | 0.8508809059734708 |  8.60455472184738 |
| *Test Set*     |  0.7996773341190241 | 9.976079452698436 |

As we can see, the difference between training and test performance has been reduced.

## XGBoost

The XGBoost Regressor is an advanced version of the Gradient Boosting Regressor that incorporates regularization techniques. From XGBoost we imported the *DMatrix* library, which offeres optimized implementation for training boosting models. The main advantage that DMatrix brings to us is that this structure allow strong data in a highly efficient format, that results in faster training times for our model. We started by defining the model and the parameters for the GridsearchCV, then after obtaining the result, we trained the model with training set and make prediction on both training and test sets.

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| *Training Set* | 0.967756117709457 |  4.0017164819033795 |
| *Test Set*     |  0.8096215795479538 | 9.719864384521323 |

As we can see, the model is overfitting again, therefore we have to reduce the complexity of the model. As before, we have changed the hyperparameters manually and more specificly we have tried these combination of hyperparameters:

- learning_rate=0.1, max_depth=7, n_estimators=300
- learning_rate=0.1, max_depth=7, n_estimators=200
- learning_rate=0.1, max_depth=7, n_estimators=100
- learning_rate=0.1, max_depth=5, n_estimators=100
- learning_rate=0.1, max_depth=3, n_estimators=100

We found the best result with this hyperparameter combination: (learning_rate=0.1, max_depth=3, n_estimators=100, random_state= 42)

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| *Training Set* | 0.8513600659849165|  8.590719216933199 |
| *Test Set*     |  0.8001676089429381 | 9.963864117889782 |

However, after launching the plot, we can see that the graph for checking the overfitting reamains almost the same.

## Random Forest

For our analysis, we have also used the Random Forest, a model that consists of multiple decision trees. To train these trees, the bagging technique is used. This algorithm iteratively evaluates a different sample with replacement of the same original training set for each tree. By doing this, we ensured that each tree in the Random Forest had a slightly different training set, resulting in a diverse ensemble of trees. We combined the predictions of these individual trees to make accurate and robust predictions for our analysis. In order to run the model we defined a parameter grid, performed a GridSearchCV and then we have calculated the performance metrics for both train and test. 

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| **Training Set** | 0.9856013458004114 |  2.6741366940457936|
| **Test Set**     |  0.8241828326387816 | 9.340755090852461 |

As we can see, also this model is overfitting, therefore we have to reduce again the complexity of the model by changing the parameters manually. The following results to be the best:

    (n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42)

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| **Training Set** | 0.8318283432158494 |   9.139007923594933 |
| **Test Set**     |  0.7725561055516224 | 10.624010406514849 |

We can say that between all the models that we have consider, the XGboost with the hyperameters optimization, is the one that performs the best reaching a good R2 score. However, as we saw from the plot for the overfitting it seems that the overall situation has not changed that much. Therefore in order to give more robustness to the predictions we have performed a Voting Regressor.

### Voting Regressor
The Voting Regressor is an ensemble machine learning algorithm that combines multiple regression models to generate predictions. It aggregates the predictions from individual regressor models and outputs either the average or weighted average prediction.

In our approach, we utilize the following models for the VotingRegressor:

1. XGBoost (the one that overfit the least)
2. RandomForest (the one that overfit the least)
3. LinearRegressor

We decide to use linear regressor rather than gradient boosting, even if the perfomance was worst, because we were motivated by the desire to capture different aspects and patterns in the data, and gradient boosting has a very similar training process to XGBoost.

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| **Training Set** | 0.8370546740686416 |  8.995879017573996 |
| **Test Set**     |  0.7751210083973059 | 8.995879017573996|

After performing the code, we can see the performance of the voting regressor is lower than our best model, therefore we can affirm that *our best model is the **XGBoost!***


# Feature importance with SHAP

SHAP is a library that leverages a mathematical method to elucidate the predictions made by machine learning models. It draws upon game theory concepts and can be employed to explicate the predictions of any machine learning model by calculating the contribution of each feature to the prediction. In our case, we applied SHAP to our top-performing model, XGBoost, to enhance its interpretability and gain a better understanding of its prediction process.

To begin, we need to create an explainer object by specifying the model and the dataset that will be utilized for making predictions. Then we can start to plot our graph. In the first graph we can see the contribution of each variable to the predictions. Higher is the shape value, higher is the importance of that predictor. However, here we have only the total importance of the specific feature without saying if it is directly or inversely correlated with the response variable. 

To continue, we have made another plot that enables to capture the direct and indirect relationship for each predictor with the value predicted.

The last plot has helped us to see how the variables contributed to the predictions. (positive and high, influence directly and a lot the prediction, negative and low, influence inversely the prediction a lot). The most important features to predict the popularity score are album_name, track_genre and artists, therefore the recommendations should be more focus on that rather than the other audio_features.

# Conclusions
To conclude our project we have provided two tables to easily see which are the mean or median values of the audio features of the most popular songs (we considered only the songs with a popularity score higher than the average).

To conclude, we want to underline something that we would have liked to implement, however due to the late discovery we were unable to complete it. In order to solve the overfitting problem, we had tried to make our features "Gaussian" or normally distributed through box-cox and logarithmic transformations but still the performance of the model did not increase.
However, we have become aware of the yeojohnson transformation too late and due to lack of time we were not able to perform the models after the transformation.
