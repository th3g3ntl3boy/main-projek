from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

k_means = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def kmeans():
    vx1 = request.form["x1"]
    vx2 = request.form["x2"]
    vx3 = request.form["x3"]
    vx4 = request.form["x4"]
    vx5 = request.form["x5"]
    vx6 = request.form["x6"]
    vx7 = request.form["x7"]
    vx8 = request.form["x8"]
    vx9 = request.form["x9"]
    vx10 = request.form["x10"]
    vx11 = request.form["x11"]
    df = pd.read_csv("Book1.csv")
    dff = df.iloc[:,2:]
    dff = dff.fillna(0)
    dfw = dff.drop(['Banyak Ternak Babi','Banyak Ternak Sapi Perah','Banyak Ternak Kelinci','Banyak Ternak Kuda','Produksi Susu Rakyat','Jumlah Pemerahan Susu'], axis = 1)
    scaler = MinMaxScaler()
    df1 = scaler.fit_transform(dfw)
    model = k_means.fit(df1)
    dfw.loc[len(dfw.index)] = [vx1,vx2,vx3,vx4,vx5,vx6,vx7,vx8,vx9,vx10,vx11]
    new = scaler.fit_transform(dfw)
    testing = new[14]
    testing = testing.reshape((1,-1))
    klasternya = str(model.predict(testing)+1)
    return render_template("index.html", myklaster = klasternya)

if __name__ == "__main__":
    app.run(debug=True)