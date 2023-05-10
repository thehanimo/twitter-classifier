from flask import Flask
from flask import request
from pymongo import MongoClient
from flask_cors import CORS, cross_origin
from flask import jsonify, Response
from flask import render_template, request, url_for, redirect
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bson.objectid import ObjectId
import subprocess
from threading import Thread
import os
import time
from classify import classify

app = Flask(__name__)

#mongo code
client = MongoClient('localhost', 27017)
db = client.flask_db
todos = db.todos
tweets = db.tweets

# model = pickle.load(open('../model.pkl', 'rb'))
# vectorizer = pickle.load(open('../tfidf.pkl','rb'))
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# @app.route("/add", methods=["POST"], strict_slashes=False)
# @cross_origin()
def predict_val(to_pred):
    to_predict = []
    to_predict.append(to_pred)
    prediction = model.predict(vectorizer.transform(to_predict))
    
    output = prediction[0]
    if int(output) == 0:
        output_result = " Is Real News"
    else:
        output_result = " Is Fake News"

    return output_result 


@app.route('/', methods=("GET", "POST"))
def index():
    if request.method=='POST':
        content = request.form['content']
        degree = request.form['degree']
        todos.insert_one({'content': content, 'degree': degree})
        return redirect(url_for('index'))

    all_todos = todos.find()
    return render_template('index.html', todos=all_todos)


@app.post('/<id>/delete/')
def delete(id):
    todos.delete_one({"_id": ObjectId(id)})
    return redirect(url_for('index'))

@app.route("/result", methods=['POST'], strict_slashes=False)
@cross_origin()
def result():
    text = request.json.get('usertext')
    predicted = predict_val(text)
    print(predicted)
    # print(type(text))
    return jsonify("The Given News '" + text  + "'" + predicted)

@app.route('/triggerHadoop', methods=["GET"])
@cross_origin()
def triggerHadoop():
    Thread(target = run_hadoop).start()
    return Response()

def run_hadoop():
    to_process = tweets.find({'label': "Pending"})
    fo= open("test.txt", "w")
    for x in to_process:
        line = str(x['_id']) + '\t' + x['text']
        line.replace('\n', " ")
        fo.write(line)
        fo.write('\n')
    fo.close()
    cmd_str = "hadoop fs -rm -r -f twitter-test.txt twitter-test"
    subprocess.run(cmd_str, shell=True)
    cmd_str = "hadoop fs -copyFromLocal test.txt twitter-test.txt"
    subprocess.run(cmd_str, shell=True)
    cmd_str = 'mapred streaming -files mapper.py,reducer.py -input "twitter-test.txt" -output twitter-test -mapper "python3 mapper.py" -reducer "python3 reducer.py"'
    subprocess.run(cmd_str, shell=True)
    cmd_str = 'rm -rf twitter-test'
    subprocess.run(cmd_str, shell=True)
    cmd_str = 'hadoop fs -copyToLocal twitter-test'
    subprocess.run(cmd_str, shell=True)
    arr = os.listdir('twitter-test')
    for filename in arr:
        with open('twitter-test/'+filename) as file:
            for line in file:
                line = line.rstrip()
                num_users = int(line.split('\t')[0])
                ids = line.split('\t')[1:num_users+1]
                tweet = "\t".join(line.split('\t')[num_users + 1:])
                tags = classify(tweet)
                if len(tags) == 0:
                    tags = ["Uncategorized"]
                for id in ids:
                    tweets.update_one({"_id": ObjectId(id)}, {"$set": {"tags":tags, "label": "Done"}})
    print("Done classifying!")
                



        


@app.route('/addData', methods=["POST", "OPTIONS"])
@cross_origin()
def addData():
    data = request.get_json()
    for x in data:
        username = x['username']
        texts = x["tweets"]
        for text in texts:
            if not tweets.find_one({'_id': ObjectId(text["_id"])}):
                tweets.insert_one({'_id': ObjectId(text["_id"]), 'username': username, 'text': text["text"], "label": "Pending"})
    return Response()

@app.route('/getTags', methods=["POST", "OPTIONS"])
@cross_origin()
def getTags():
    data = request.get_json()
    for x in data:
        username = x['username']
        texts = x["tweets"]
        for text in texts:
            curr = tweets.find_one({'_id': ObjectId(text["_id"])})
            if not curr:
                continue
            text["label"] = curr["label"]
            if "tags" not in curr:
                continue
            text["tags"] = curr["tags"]
    return jsonify(data)
        

if __name__ == "__main__":
    app.run(debug=True)

