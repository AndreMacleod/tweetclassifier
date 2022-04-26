# tweetclassifier
## A model to automatically distinguish between Musk and Biden tweets

The code is easily changeable to any two users. 
Currnetly set up as an app that gives a response to localhost, given a user query. To test model, run app.py and then in the terminal (for Windows)
curl -X GET -H "Content-type: application/json"  -H "Accept: application/json"  -d "{\"query\":\"TEST TWEET\"}"  "http://127.0.0.1:5000/"
Trying 'I love bitcoin' and 'we stand with Ukraine' gave Musk and Biden responses, as a very small initial test. The model itself has a test set, and the resulting CM is included as png. Metrics are as follows:

Evaluation metrics

              precision    recall  f1-score   support

       biden       1.00      0.73      0.84        11
        musk       0.87      1.00      0.93        20

    accuracy                           0.90        31
   macro avg       0.93      0.86      0.89        31
weighted avg       0.92      0.90      0.90        31

## Potential Improvements

The model could be made to distnguish tweets from any 2 people or potentially more. Maybe not between users but topic, perhaps a way to distinguish between left and right political tweets and have it classify those. 
Would make it into a full web app where a user can send in a link to a tweet and model gives response. Could give user chance to say if it was correct, and log this response in order to collect additional data


