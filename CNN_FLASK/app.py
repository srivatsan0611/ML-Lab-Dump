#%%

from flask import Flask, render_template, redirect, request, url_for

import numpy as np

app = Flask(__name__, template_folder="./",static_folder="./",static_url_path="")

from tensorflow import keras

model = keras.models.load_model("CNN_Model.h5")
# %%
@app.route('/')
def main():
    return render_template("index.html")

@app.route('/predict',methods=["POST","GET"])
def predict():
    file = request.files["file"]
    import imageio as ie

    pic = ie.imread(file)
    pic = pic.reshape(1,32,32,3)
    prediction = model.predict(pic)
    prediction = [np.argmax(p) for p in prediction]
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    final = classes[prediction[0]]

    return render_template("result.html",prediction=final,image=file.filename)

# %%
if __name__ == "__main__":
    app.run(host='localhost',port=5000,debug=False)
# %%
