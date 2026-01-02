# Reddit Rule Violation classification
Predict whether a subreddit comment violates the rule of that given subreddit.

## Problem Statement:- 
Analyse ~3k moderated comment records from subreddits to build a NLP model that:
* identifies and flag comments whenever rules are violated.
* Help automate moderation.

## Dataset
- __Source__: [Kaggle](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
- __Records__: 2029
- __Target__: `Rule Violation`(binary)

## Tech Stack
* __Core Stack__: Python, MongoDB, Pandas, NumPy
* __Deep Learning__:  PyTorch, Transformers, Sentencepiece
* __MLOps & Deployment__:Docker, FastAPI, AWS
* __Automation__: GitHub Actions (CI/CD)

## Key Challenges
* Selecting proper method for clusturing rule-violation.  
* Methods of text encoding that captures semantics and keeps size down. 
* Modifying architecture for sizing down the dockerfile.

## Evaluation Strategy
- ROC-AUC used as primary metric due as it captures model's potential to seperate `harmful` from `harmless` comment.
- F1_score was used as secondary metric as it emphasizes both maintaining the quality of predictions(precision) and high number of true rule violations (recall).



| Model | roc_auc_score | Recall Score | Accuracy Score | F1 Score | Precision 
|-------|------|------|------|------|------|
|PyTorch|0.9419|0.9214|0.9417|0.9412|0.9618|


## Limitations
- Noisy and informal Reddit text can reduce semantic accuracy.
- The two-layer neural network has limited capacity for complex language patterns.
- Comments are classified without conversational context.
- Small dataset size may limit generalization.


## Future Prospects 
* Generalization to unseen rules and comment pairs for more communities. 
* Introduce comprehensive test cases, edge-case handling to ensure reliability in real-world deployment.



