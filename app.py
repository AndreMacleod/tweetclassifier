
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import joblib
from script import preprocess_text
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
#from sklearn.pipeline import Pipeline

app = Flask(__name__)
api = Api(app)

# load the model
model = joblib.load('model.x')

#docs_new = ['God is love climate jbgnfrebgno enrgoinrg 4et3wt4e 4et4et']
#y_new = ['climate']
#pred= model.predict(docs_new) 
#np.array(docs_new)

# print report
print('Evaluation metrics')

#cm = metrics.confusion_matrix(y_true=y_new, y_pred=pred, labels=["climate", "business"])

#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["climate","business"])
#disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='vertical')
#plt.show()

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')
@app.route("/getjson")


class PredictTweet(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        uq_vectorized = (np.array([user_query]))
        prediction = model.predict(uq_vectorized)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction == "musk":
            pred_text = 'musk tweet'
        else:
            pred_text = 'biden tweet'

        # create JSON object
        output = {'prediction': pred_text}
  
        return output

api.add_resource(PredictTweet, '/')

# Setup the Api resource routing here
# Route the URL to the resource
#api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)


# use model to predict
#y_pred = model.predict(X_test)



# Run following in terminal for prediciton
# curl -X GET -H "Content-type: application/json"  -H "Accept: application/json"  -d "{\"query\":\"we stand with ukraine\"}"  "http://127.0.0.1:5000/"

