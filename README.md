# Popularity prediction
## Bag-of-words analysis in /r/AskReddit
### The attached code collects the titles from the top posts in /r/AskReddit using the Reddit API

Thoughts:
- Apply logistic regression to predict whether or not the post will exceed a given cutoff of upvotes)

Popularity Prediction:
  - pullData_popular.py: extracts the top posts in /r/AskReddit using the Reddit API and randomly assigns it to training or testing; assigning popularity value
  based on a predetermined cutoff value
  - model_popular.py: trains a logistic Regression model using a bag-of-words model to predict post popularity
  - popPredict.pkl: popularity model
  - rocPop: computed ROC curve for this experiment
  - Results: Essentially as good as a fair coin flip

Future considerations
  - Training set is imbalanced? Resampling or sample replication to balance
  - Different feature selections should be considered
