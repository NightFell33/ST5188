# Projects Introduction
The project did training based on the Goemotions dataset 
through LSTM and BERT methods and predicted the sentiment 
distribution of Apple product review data obtained from Reddit 
based on the results, 
and used LDA topic modeling methods to further parse the results.

# Environmental dependency
The vast majority of the code can be run under 3.12.7 
and the latest torch environments. 
Note that the LSTM training part needs to be run under a tensorflow-gpu environment.

# Introduction for folders and files

### praw_data/

This folder is for getting data from Reddit. (Note that this is the starting point for all data sections, 
other open source datasets used in this project such as Goemotions are available from the web.

apple_new.py -- file for get data from Reddit. 
Note that you may need API account to login, 
and I have put my own account in the file. 
Also note that due to the time change, 
you will get different data if you use this and following files.

3.11get_new_posts.py -- file for get new data from Reddit before 3.11.

3_11_concat_comments.py -- file to concat old and new comments.

### mid_plot/

This folder is to provide the mapping required for the midterm report.

wordcloud.py -- file for wordcolud for Goemotions dataset.

reddit_vader.py -- file for VADER sentiment analysis of Reddit comments, 
and generate wordcloud for reddit comments.

goemotions_view.py -- file for plotting the sentiment distribution of Goemotions dataset.

comments_summary.py -- file for plotting the summary of Reddit comments amounts.

### LSTM/

This folder is for LSTM model training and prediction.

read_predict.py -- This file is no actually use for the model training, 
but just have a look at the results of the model. 
You can omit it when you want to run the model.

Data_preprocessing.py -- file for data augmentation and preprocessing.

lstm_model_final.py -- file for training the LSTM model and predicting the results. 
Note that this part need a diffrent environment to run, which is based on tensorflow-gpu.

### BERT/

This folder is for BERT model training and prediction.

BERT.py -- file for model training, you can choose what kind of model you want to train

Tranfser.py -- file for tranfer learning of our model on two benchmarks, including Tweeteval-Sentiments and Stanford Sentiment Treebank

Inference.py -- file for utilizing our model on data inferencing, label the unlabeled scrawled data.

### LDA/

This folder is for LDA model training.

lda_1.py -- file for preprocessing, training LDA Model, and plot the LDA results.

### final_plot/

This folder is an additional and final drawing 
based on the results obtained earlier.

time_plot.py -- This document is a statistical distribution of the frequency of occurrence 
of 25 given word groups over time.

time_plot2.py -- Plot a line graph of the change in sentiment over time 
corresponding to the 25 given word groups.



