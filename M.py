from flask import Flask, render_template, request,redirect,url_for
from tensorflow.keras.models import load_model
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import pickle
import random
import json
import os
bb = []
stemmer = LancasterStemmer()
words = pickle.load(open("words.pickle","rb"))
labels = pickle.load(open("labels.pickle","rb"))
with open("intents.json") as file:
    data = json.load(file)
model = load_model("model.model")
def write_json(new_data, filename):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["intents"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)
def write_jsonpatres(tag,filename,pattern,response):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        for i,intent in enumerate(file_data["intents"]):
            if intent["tag"]==tag:
               intent["patterns"].append(pattern)
               intent["responses"].append(response)
        file.seek(0)
        json.dump(file_data, file, indent = 4)                 
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    b = bag
    for i in b:
        bb.append(i)
def chat(txt):
        bag_of_words(txt, words)
        print(bb)
        results = model.predict([bb])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        bb.clear()
        return random.choice(responses)
app = Flask(__name__)
username = ['None']
usernames =['Majd Khalifeh']
passcodes = ['M007007m']
@app.route('/', methods=["POST", "GET"])
def login():
    if request.method == "POST":
        textt1 = request.form['textt1']
        username.insert(0,textt1)
        textt2 = request.form['textt2']
        for i,x in enumerate(usernames):
            if x==textt1:
                if passcodes[i]==textt2:
                    return redirect('/M.A.K.P.W.S')

        return render_template("login.html")
    else:
        return render_template("login.html")
@app.route('/M.A.K.P.W.S' ,methods=["POST", "GET"])
def my_form_post():
    if request.method == "POST":
        text = request.form['text']
        res = chat(text)
        return render_template("background.html",content=res)
    else:
        return render_template("background.html",content="responding")    
@app.route('/M.A.K.P.W.S/retrain', methods=["POST", "GET"])
def my_form():
    if request.method == "POST":
        pat = request.form['text1']
        res = request.form['text2']
        tag = request.form['text3']
        print(pat,res,tag)
        if tag not in labels:
            datatoapped = {
                    "tag": tag,
                    "patterns": [
                        pat
                    ],
                    "responses": [
                        res
                    ],
                    "context_set": ""}   
            write_json(datatoapped,"intents.json")     
        else:write_jsonpatres(tag,"intents.json",pat,res)   
        os.system("train_model.py")
        return render_template("addmodel.html")
    else:
        return render_template("addmodel.html")        
if __name__ == "__main__":
    app.run(debug=True)