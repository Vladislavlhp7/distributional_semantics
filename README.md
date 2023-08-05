# Distributional Semantics
## Method Description
The task involves analyzing product reviews and implementing distributional semantics to understand the relationships between words. The following methods were used:

1. Text cleaning & pre-processing: The text data contained unhelpful characters and symbols, such as brackets, encoding strings, currency expressions, and punctuation. These were removed along with sentiment scores and details. Contractions were expanded based on predefined rules. Tokenization was performed using NLTK's recommended word tokenizer, and stop words were removed to reduce document size. Snowball Stemmer was applied for word stemming.

2. Distributional semantic representations: After cleaning the text data, it was partitioned to a sentence level using '##' and '\n'. The word2vec method was chosen to represent sentences, allowing modeling based on word neighbors. Both CBOW and skip-gram models were experimented with.

## Result Analysis
The hyperparameters were evaluated to optimize the performance:

* Number of epochs: [5, 10, 30, 50, 75, 100]
* Word2vec dimensions: [100, 164, 200, 300]
* Window sizes: [2, 3, 4, 5]
* CBOW or Skip-gram models

The best model selected was Skip-gram trained for 30 epochs with an accuracy of 82% and 100 feature dimensions with a window size of 2.

# Neural Network for Classifying Product Reviews
## Method Description
The task involves classifying product reviews into positive, negative, or neutral sentiments using a neural network. The following methods were used:

1. Text cleaning & pre-processing: Sentiment scores were unified, and qualitative sentiments were excluded. A Sentiment Extractor was used to parse review-level and sentence-level sentiments, creating a data structure with binary sentiment scores.

2. Model design & training: The data was split into train and test datasets, and a bidirectional LSTM model was implemented with global max pooling and dense fully-connected layers. A dropout layer was added for regularization.

## Result Analysis
Hyperparameter tuning was performed to optimize the model:

* LSTM dimensions: [32, 64, 128, 180, 256]
* Fully-connected final layer size: [10, 50, 100]
* Dropout rate: [0.01, 0.1, 0.3]
* Training epochs: Up to 10 (later limited to 3)

The best hyperparameters were identified, and the model was trained using both K-Fold and Stratified cross-validation techniques. The results showed that minimal hyperparameters were sufficient for the model's accuracy.

## Optimization Suggestions
* Use word2vec for text representation when passing text to the DL's input layer.
* Enhance the architecture with a convolutional layer to extract text features and reduce dimensionality.
* Embed sentiment word scores as word weights to review-level representations for differentiation between positive and negative words.
* Implement a self-attention mechanism or a transformer-based architecture for small training data.
