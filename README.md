# spell-check-LSTM
The objective of this project is to build a model that can take a sentence with spelling mistakes as input, and output the same sentence, but with the mistakes corrected. The data that we will use for this project will be twenty popular books from [Project Gutenberg](http://www.gutenberg.org/ebooks/search/?sort_order=downloads). Our model is designed using LSTM(Keras).


# TECHNOGIES 
Keras , Python, NLP , Sklearn, Numpy, Os 

# Project section:
Our project is basically divided into four section:
    1. LOADING DATA 
    2. PREPARING DATA 
    3. BUILDING MODEL 
    4. TRAINING MODEL 

# STEPS INVOLVES:
    1. First insert the dataset from the respective path, Here I use 18 books as dataset.
    2. After inserting dataset first remove unwanted character from the corpus.
    3. After that converting the character into integer according to occurance of the character in corpus
    4. After that train and test split of data take place by using train_test_split
    5. Further from train split data I get extract the Ground  Truth , noise data 
    6. By using noise function convert the  train data into noise data 
    7. Then further padding for all different dataset take place
    8. Further building the LSTM by using Keras model 
    9. Insert data into the deep learning model 
