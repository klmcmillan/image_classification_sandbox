# image_classification_sandbox

This repository provides scripts to create unique image datasets and explore different algorithms used for image classification. Usage instructions and descriptions of the various files are provided below.

## <a name="download"></a>download_images.py

This file can be used to create image datasets based on user-specified web searches using the Bing search engine. In order to use this file, the following lines from the ```main()``` function need to be changed to refleect the desired image searches:

```python
save_dir = 'images'
image_type = 'food'
queries = ['tacos', 'pizza', 'pasta', 'hamburgers', 'bbq']
```

```save_dir``` is the directory where the images will be saved. The directory will be created unless it already exists. ```image_type``` is the file name that will be used when saving the set of related images. For example, if you wanted to download a set of images related to "food", you would use ```image_type = 'food'```. Images would then be saved with the filenames "food_1", "food_2", etc. ```queries``` are the search terms related to ```image_type```. So if ```image_type = 'food'```, search terms of interest might be "tacos", "pizza", "pasta", etc. The code will iterate through the list of search terms and return images that result from Bing image searches using those search terms.

After changes to the ```main()``` function are made, run the file, and images for each of the search terms will be saved in the user-specified directory. This is a very bare-bones image search process that does not take advantage of the [Bing Image Search API](https://www.microsoft.com/cognitive-services/en-us/bing-image-search-api). For each search term, only 28 images are returned. Therefore, if you provided 5 search terms, you would get 140 images.

Because the name of the game is image classification, this file should be run multiple times with different ```image_type```. Maybe you want to see how well an image classifier works on a mix of images of "food" and "people". Simply run download_images.py once with ```image_type = 'food'``` and ```queries``` related to "food" and then then run it a second time with ```image_type = 'people'``` and ```queries``` related to "people". Each time you execute download_images.py, keep ```save_dir``` the same so that images for all ```image_type``` are saved in the same directory.

## classify_images_knn.py

This file uses the [k-nearest neighbors (KNN) algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) to classify images from a dataset created using [download_images.py](#download).

### Pre-processing

The following pre-processing steps are taken to prepare the image data for input into the KNN algorithm:

1. Transform the 2D images into a 1D arrays of RGB pixels
2. Reduce the dimensionality of the feature space through the use of [principle component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)

#### Transform images

In any classification problem, features need to be determined from the data. One way to transform images into a format that can be used within a classification algorithm is to convert the images into a set of RGB pixels. Each image is saved as a (m x n) array of pixels. This 2D array can be flattened into a (1, m x n) array. Most of the images downloaded from Bing have a standard array size of (230 x 170) pixels. The flattened version of this array is 39,100 RGB pixels. If we stopped here, each image would be represented by 39,100 features!

#### Dimensionality reduction

39,100 features is far too many features to input into the classification algorithm. One way to reduce the feature space is to use PCA. The ultimate goal of PCA is to explain the maximum amount of variance in the data with the fewest number of features (principle components). In this code, the user can explictly set the number of principle conponents that should be used. For visualization of the data, the user would set the number of principle components equal to 2.

### File usage

In order to use this file, the following lines from the ```main()``` function need to be changed:

```python
data_dir = 'images'
test_size = 0.4
n_components = 5
n_neighbors = 15
n_mesh = 150
```

```data_dir``` is the directory where the image data is saved. ```test_size``` is the fraction of total image data that should be categorized as testing data. For example, if ```test_size = 0.4```, 40% of the image data will be used as testing data and 60% will be used as training data. ```n_components``` is the PCA set of dimmensions. ```n_neighbors``` is the KNN number of neighbors. ```n_mesh``` is the number of elements in the meshgrid. This parameter is used for adjusting the visualization of the classification decision boundaries.

After changes to the ```main()``` function are made, run the file, and the following will happen:

1. Each image will be transformed into a 1D array and saved to a [sklearn-like dataset](http://scikit-learn.org/stable/datasets/)
2. The data will be split into training and testing data
3. The feature space of the training and testing data will be reduced to ```n_components``` number of features
4. A [KNN classifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) will be trained using the training data
5. The trained KNN classifier will be applied to the testing data, and performance metrics will be printed to screen (example provided below)
6. A plot of the decision boundaries for each class will be generated (example provided below)

### Example

As an example, I created a dataset of images of common electronics (e.g. computers, cell phones) and food (e.g. pizza, tacos) using multiple instances of download_images.py. This data can be found [here](https://github.com/klmcmillan/image_classification_sandbox/tree/master/images). After running classify_images_knn.py, the following performance metrics were printed to screen:

![Classifier performance metrics](https://github.com/klmcmillan/image_classification_sandbox/blob/master/examples/knn_metrics.png)

Additionally, the following plot of the decision boundaries for the "electronics" and "food" classes was generated:

![Decision boundary visualization](https://github.com/klmcmillan/image_classification_sandbox/blob/master/examples/knn_classification.png)
