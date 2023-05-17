# Popularity score of music tracks
- Michele Tresca 
- Greta Kichelmacher 
- Silvia Conti

# Introduction 
Due to the rising popularity of digital music platforms, understanding the characteristics that contribute to a song's popularity has become increasingly important, not only for music producers and artists but also for the platforms themselves. Thanks to the  dataset provided to us, we were able to perform an in-depth analysis of the factors influencing a music track's popularity. Based on our findings, we aim to offer valuable recommendations for music producers and artists.


#  EDA
Firstly, we import a selection of libraries that will facilitate our data analysis:

- **"sklearn"**: This library offers a broad array of machine learning algorithms, tools for data preprocessing, and functions for performance evaluation.
- **"numpy"**: It's a powerful tool for scientific computation in Python.
- **"pandas"**: This library provides easy-to-use data structures and data analysis tools.
- **"matplotlib"** and "seaborn": Both libraries are used for data visualization, with Seaborn being built on top of Matplotlib and offering a higher level of abstraction for statistical plots.

Next, we import our CSV file and read the data into a DataFrame. Using the *`*df.head()*`* function, we can quickly display the initial rows of our DataFrame, giving us a glimpse into our dataset. 

To better understand the structure and type of our data, we use the *`df.info()`* function. This helps us identify the data types for each of the columns in our DataFrame. Here's what we find:

- The columns Unnamed, popularity, duration_ms, key, time_signature, and mode are of integer type.
- `track_id`, `artists`, `album_name`, `track_name`, and `track_genre` that are objects, specifically, they are strings.
- Explicit is a boolean value.
- `danceability, energy, loudness, speechiness, acousticness, instrumentalness, livness, valence,` and `tempo` are float values.

This breakdown gives us a clear understanding of our dataset's structure and the types of data it contains.
Using the `df.shape` function, we ascertain the dimensions of our DataFrame, which contains 114,000 rows and 21 columns.

Next, we employ the `df.describe()` function to generate summary statistics for our data, which includes *mean, standard deviation (std), minimum (min), and maximum (max) values*. From this summary, we observe that the mean popularity is 33%, with a standard deviation of 22%. The minimum popularity is 0 and the maximum is 100. This distribution suggests a leftward skew in the data, indicating that the majority of elements have low popularity, with only a few outliers enjoying very high popularity.

We proceed to check for invalid values and discover *NaN (Not a Number*) values in the columns: *artists, album_name, and track_name.* Since these are string-type columns, we don't opt to replace these missing values with mean or median values as it wouldn't be meaningful. Considering that these NaN values represent only 3 rows out of the total 114,000, we decide to remove these rows. The loss of data is inconsequential in this context. By specifying **`inplace=True,`** we ensure that our function directly modifies the original DataFrame, effectively removing rows with missing values and overwriting the existing DataFrame.

Following this, we remove the first column labeled *'Unnamed'* as it merely duplicates the index column, offering no new information.

Lastly, we check for and eliminate any duplicates within our DataFrame. By using the `df.duplicated().sum()` function, we find that there are 450 duplicate entries. We proceed to remove these using the df.drop_duplicates **`(inplace=True)`** function, ensuring our data is free from redundancies.

We proceed to examine the correlation between variables by creating a correlation matrix. To make these correlations visually intuitive, we plot a heatmap which uses a color gradient to highlight the strength of correlations. Notably, the variables `valence`and `danceability` exhibit a strong correlation, as do `'loudness'` and `'energy'.` Interestingly, we observe that popularity does not have a direct or linear correlation with any other parameters.

By employing the `sns.pairplot(df)` function, we generate a scatter plot matrix, which visually represents the relationships between all pairs of variables in the DataFrame. This further confirms the absence of linear correlations. 

Subsequently, we plot the distributions of each parameter to gain a deeper understanding of our data. For instance, we find a significant number of zeroes in the `popularity` distribution. From the 'duration' histogram, we observe that while the values range from 0 to 5, they are primarily concentrated between 0 and 1. As for the `mode` histogram, since it's a boolean variable, the values are either 0 or 1.

Next, we create a box plot for selected variables termed `audio_features`, which share a common range of values between 0 and 1, and are hence comparable. From this plot, it's evident that the variables `danceability, energy, acousticness`, and `valence` demonstrate reasonable values, while other variables exhibit a significant number of outliers. Particularly, we notice a high concentration of data at lower values, with outliers at higher values. We then generate individual box plots for `Tempo` and `Loudness`.

Although we've previously plotted scatter plots for all variables, we create additional scatter plots for a select few.

We then delve into the top 10 music genres in terms of their average popularity. By calculating the average popularity score for each music genre, we group the DataFrame rows by genre. We then compute the mean and standard deviation of popularity for each genre and plot a histogram that associates the top 10 music genres with their popularity. This histogram reveals that the music genre with the highest average popularity (pop-film) has a lower standard deviation in comparison to another music genre (pop), which, despite its lower average popularity, has a higher standard deviation. We repeat this process for the top 10 artists.

Lastly, we use `df[df['track_id'].duplicated()].sort_values('track_id')` to sort the DataFrame by 'track_id' and identify any duplicated entries.

# Data Preprocessing
We had a lot of information  about  songs,  but  some of it  was  repeated. We thought that if we only  kept the most popular version of each song, we could learn more about popular songs. 
But we also thought  it  would  be  helpful to know  how  different  types of music  affect  how  well a song  does. 
By having this in mind, we provide the code for data preprocessing that can be changed between True or False depending on the idea for the analysis. Moving forwrard the 'explicit' column is converted to int64 to ensure compatibility with machine learning algorithms that require numerical inputs. Then we've created a new DataFrame called `X` that includes all the remaining columns from `df`, and then we've removed `popularity, track_id, track_name` from `X`. 
This DataFrame will serve as the input for training the machine learning model.  Lastly  `y` is created, containing only the `popularity` column from `df`, representing the target variable that the model aims to predict. 

### Data Binning
As the distribution of the popularity score was not constant among different values we implemented a data binning methodology to partition the original dataset into a training and test set containing the same proportion of popularity Score values. We started by creating creating bins and labels to categorize the values in the `y` column into distinct categories. Then, a new column 'bins' is added to the DataFrame `X`, containing the assigned category for each value in `y`. Then we've splitted the data into training and testing sets, maintaining the class distribution based on the `bins` column. 
The feature and target variable data are assigned to separate variables for both the **training and testing sets**. Then we've removed the `bins` from both the training and testing DataFrames. Moving forward we've decided to make some plot to make some analysis, and from that we've noticed that some Values above the *99th percentile* threshold could be seen as noise, se we decided to replace it with the threshold value itself. 
This ensures that extremely high values do not disproportionately influence the model's training process.

### Encoding
considering that the variable:
- `key` can assume only integer values ranges from 0 and 11
- `time_signature` only integer values ranges from 0 to 5 

We've decided to treat these two features as categorical variables based on the meaning of the features and therefore we decided to encode them with the One Hot Encoding. After the OHE, we've decided to use the Leave One Out Encoder, that's because considering the general importance of the artist, the genre of the track and the name of the album, including this features inside our model will be very important.
After the encoding we've performed a code to check the correlation with the features that are not anymore categorical. According to the analysis, we observe that the variables 'track_name,' 'album_name,' and 'track_genre' exhibit significant correlation with the target variable. Additionally, there is a strong correlation between 'album_name' and popularity. However, to retain the meaningful information for distinguishing popularity levels based on 'album_name,' it is advisable not to drop this variable (by setting 'drop_variable=False').

### Feature Scaling 
We performed the feature scaling in order to standardize or normalize the numerical features in a dataset to ensure that they are on similar scale. Then we made a code that separates the columns containing dummy variables or categorical features from the rest of the encoded data. These columns are extracted into new DataFrames (`X_train_cat` and `X_test_cat`), while the original DataFrames (`X_train_encoded` and `X_test_encoded`) have these columns removed. This dummy variables separation is important because those kind of variables have a different scale (0 or 1) compared to continuous variables. By using feature scaling, specifically the **StandardScaler,** we have achieved some benefits like:

- Normalization of the distribution
- Handling the outliers
- Mantaining the interpretability of the data
- Ensuring equal importance for all features

Lastly, After performing the standard scaler, we've made a code that creates sequential identifier columns (`merge` and `merge1`) for the rows in the scaled training and testing DataFrames. 
These columns are then used to merge the categorical and numerical feature DataFrames with the scaled DataFrames, respectively. 
The resulting merged DataFrames contain both the categorical and numerical features, while the identifier columns are subsequently dropped.

# Clustering
To perform the task related to cluster analysis We've started by creating a new dataframe called `df_cluster`, which is an exact copy of the `X_train_scaled` dataframe. Then, we've setted  the number of unique music genres as the number of clusters, aiming to see if *K-Means* could identify groups of songs that correspond to music genres. Lastly, We created a new column called `cluster`, where we inserted the cluster membership of each song into the dataframe

Then we've performed the *accuracy score*.
Our analysis revealed that **the model's accuracy is very low (0.0001425438596491228),** indicating that the clusters identified by KMeans do not closely correspond to the original music genre labels. This **is not surprising**, given that KMeans is an unsupervised clustering algorithm and does not use labels during training. Additionally, music genres can be quite subjective and might not directly correspond to measurable audio features of a track.

However, we also computed the *Normalized Mutual Information (NMI*) between the clusters and the original labels, **obtaining a score of 0.4019656340208907**. NMI is a measure that quantifies the association between two categorical variables, and can be a better option than accuracy to evaluate clustering results. The relatively high NMI score suggests that there is some association between the clusters and the music genres, although it's not perfect.

In conclusion, while using KMeans to infer a track's genre based solely on its audio features can yield some interesting results, it doesn't appear to be a highly accurate technique. This could be due to the subjective nature of music genres, which might not be fully captured by audio features."

# Models
Before starting to perform the models we've imported some sklearn models and metrics that are very important. 
For every model, the code will perform model evaluation by calculating R^2 and RMSE values for both the training and test sets. 

### Linear Regression
Before calculating the performance metrics on data, we had to create the linear regression model and to fit it on X_train_scaled, y_train. These metrics provide insights into the predictive performance of the model, indicating the proportion of variance explained by the model and the average prediction error. Those are our results:

|              | R^2              | RMSE               |
|--------------|------------------|--------------------|
| **Training Set** | 0.7873274350786287 | 10.275844108326021 |
| **Test Set**     | 0.7152642774777251 | 11.893668564220878 |

(We have calculated the train and test for all the models, but from now on we will show only our test results)
Even if the relationship between the features and popularity was not linear, we can see that the results are not bad, and this can due to the fact that we added some linearity with the categorical variables encoded with LeaveOneOut that we were not able to see previously.
Now we're gonna perform some scatter plot analysis for `track_genre, artists` and ` album_name.`
The three scatter plots allow for exploring the relationship between `'y_train2'` and specific predictor variables, enabling the identification of potential correlations or patterns that can be useful for subsequent analysis and modeling. From thios graphics We can see that the relationship between album_name and popularity, and between artists and popularity is a little bit more linea
