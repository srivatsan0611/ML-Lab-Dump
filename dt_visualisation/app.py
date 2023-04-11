from flask import Flask,render_template

app = Flask(__name__,template_folder="./",static_folder="./",static_url_path= "")

import pickle

model = pickle.load(open("model.pkl","rb"))


@app.route("/",methods = ["GET","POST"])
def hi():
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
    cn=['setosa', 'versicolor', 'virginica']
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    plot_tree(model,
    feature_names = fn, 
    class_names=cn,
    filled = True)
    fig.savefig("dt.png")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="localhost",port=5000)