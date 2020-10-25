# Machine Learning Challenge - Exoplanets

## Objective

This objective of this exercise is to utilize machine learning methods, specifically the [Scikitlearn](https://scikit-learn.org/stable/) library in Python, to solve a classification problem of stellar proportions (pun intended). The approach makes use of Numpy, Pandas, Scikitlearn, and [TensorFlow](https://www.tensorflow.org/) via [Keras](https://keras.io/guides/sequential_model/).

## Background

The [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/data.html) hosts a repository of [exoplanet](https://en.wikipedia.org/wiki/Exoplanet) data collected by the [Keplar Space Telescope](https://en.wikipedia.org/wiki/Kepler_space_telescope). The telescope was in use for over nine years from March 7, 2009 through its retirement on October 30, 2018. Over that span, the telecope observed over half a million stars and detected over 2500 exoplanets. 

Planetary data collected by various means (including the telescope) has been used to classify Keplar objects of interest (KOIs) as confirmed exoplants, candidates, and false positives. Classifications are assigned based on over two dozen characteristics/features. A full table of the data, including definitions for each of the features, can be found at the archive [here](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

## Approach 1 - Random Forest Classifier

The [random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) was selected as the first classification approach due to its robustness. Outline of steps:
1. The data was pre-processed into usable form for the random forest algorithm
    - The data from the CSV file was imported using Pandas
    - All columns from the data frame except for 'koi_tce_plnt_num' and 'koi_disposition' were assigned as features (`X`)
    - The disposition column of the data frame (`koi_disposition`) was assigned as the `y`.
    - The data was split training and test sets using `train_test_split`.
    - The training and testing feature data (`X_train` and `X_test`), all of which was numeric, was scaled using `MinMaxScaler`, creating the scaled parameters `X_train_scaled` and `X_test_scaled`.
2. The `RandomForestClassifier` was invoked and fitted to the scaled X training and y training data with 200 trees being constructed for the forest
3. The model was tested against the test data
4. The `feature_importances_` property of the classifier was used to identify the most influential features of the the dataset, only the top 4 of which had importances of over 10%. The remaining features were below 3% importance. 
5. The model was re-tuned using the top 10 most important parameters
6. The re-tuned model was then subject to hyperparameter tuning using the `GridSearchCV` function. The following parameters and levels were selected for exploration (explanation of the parameters can be found in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforest#sklearn.ensemble.RandomForestClassifier)):
    - n_estimators: 200, 400
    - min_samples_split: 2, 4, 8
    - min_samples_leaf: 4, 8, 12
    - max_features: None, 'auto'
7. Following grid fitting, the re-fitted model was tested against the test data with the recommended parameters.
8. The model was saved using `joblib`.

### Results

## Approach 2 - Deep Learning Neural Network

The deep learning neural network is a power, if not quite well understood, method for classification. It was selected here due to the ease of adding additional layers to make the algorithm more powerful. Outline of steps:
1. The data was pre-processed similarly to the Random Forest approach above. However, the [`Sequential` model](https://keras.io/guides/sequential_model/) in Keras requires `y` to be encoded as a binary numpy array. This was accomplished by using the `LabelEncoder` from Scikitlearn to convert the `y_train` and `y_test` categorical data into compatible arrays.
2. The initial sequential neural network model was constructed with a single hidden layer of **100** nodes. There were **39** input features and three outputs (one for each of the three types of binary arrays representing the categorical exoplanet classification).
3. The model was fit to the X and y training data with shuffling and 100 epochs.
4. The fit model was evaluated against the test data.
5. Based on the results of steps 3-4, a new model was constructed with two additional hidden layers of 100 nodes each. The model was fit to the same training data with shuffling and 100 epochs.
6. The model as saved using the keras `save` method for Sequential models.

### Results

## Approach 3 - K Nearest Neighbors

As a more tangible comparison to the random forest, a model was also constructed using the [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.radius_neighbors) algorithm. Outline of steps:
1. The data was pre-processed similarly to the Random Forest approach above
2. The `KNeighborsClassifier` was invoked and executed on the training and testing data for `n_neighbors` from 1 to 100 counting by 2. For each iteration the training and testing scores were calculated using the `.score` method on the classifier.
3. Once an optimal *k* (n_neighbors) was selected from step 2, hyperparametric tuning was performed in an attempt to increase the model's accuracy, again using `GridSearchCV`. The following parameters and levels were selected for tuning (explanation of the parameters can be found in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors))
    - algorithm: 'ball_tree', 'kd_tree', 'brute'
    - leaf_size: 10, 30, 60, 90, 120
    - p: 1, 2
4. The retuned model was evaluated against the test data with with the recommended parameters.
5. The model was saved using `joblib`.


### Results

