# image_classification_sandbox

This repository provides scripts to create unique image datasets and explore different algorithms used for image classification. Usage instructions and descriptions of the various files are provided below.

## <a name="download"></a>download_images.py

This file can be used to create image datasets based on user-specified web searches using the Bing search engine. In order to use this file, the following lines from the ```main()``` function need to be changed to refelect the desired image searches:

```python
save_dir = 'images'
image_type = 'food'
queries = ['tacos', 'pizza', 'pasta', 'hamburgers', 'bbq']
```

```save_dir``` is the directory where the images will be saved. The directory will be created unless it already exists. ```image_type``` is the file name that will be used when saving the set of related images. For example, if you wanted to download a set of images related to "food", you would use ```image_type = 'food'```. Images would then be saved with the filenames "food_1", "food_2", etc. ```queries``` are the search terms related to ```image_type```. So if ```image_type = 'food'```, search terms of interest might be "tacos", "pizza", "pasta", etc. The code will iterate through the list of search terms and return images that result from Bing image searches using those search terms.

After changes to the ```main()``` function are made, run the file, and images for each of the search terms will be saved in the user-specified directory. This is a very bare-bones image search process that does not take advantage of the [Bing Image Search API](https://www.microsoft.com/cognitive-services/en-us/bing-image-search-api). For each search term, only 28 images are returned. Therefore, if you provided 5 search terms, you would get 140 images.

Because the name of the game is image classification, this file should be run multiple times with different ```image_type```. Maybe you want to see how well an image classifier works on a mix of images of "food" and "people". Simply run download_images.py once with ```image_type = 'food'``` and ```queries``` related to "food" and then then run it a second time with ```image_type = 'people'``` and ```queries``` related to "people". Each time you execute download_images.py, keep ```save_dir``` the same so that images for all ```image_type``` are saved in the same directory.

## classify_images_knn.py

This file uses the [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) to classify images from the dataset you created using [download_images.py](#download). In order to use this file, the following lines from the ```main()``` function need to be changed:

```python
data_dir = 'images'
test_size = 0.4
n_components = 5
n_neighbors = 15
n_mesh = 150
```
