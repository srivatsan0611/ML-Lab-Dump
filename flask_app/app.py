from flask import Flask,render_template,redirect,request
import pickle

app = Flask(__name__,template_folder="./")

with open("model.pkl","rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    i1 = float(request.form["SepalLengthCm"])
    
    i2 = float(request.form["SepalWidthCm"])
    
    i3 = float(request.form["PetalLengthCm"])
    
    i4 = float(request.form["PetalWidthCm"])

    input = [i1,i2,i3,i4]

    prediction = model.predict([input])
    print(prediction)

    return render_template("result.html",prediction = prediction)

if __name__ == "__main__":
    app.run(host="localhost",port=5000)