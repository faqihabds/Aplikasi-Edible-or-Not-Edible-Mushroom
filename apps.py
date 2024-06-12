import os
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

dic = {0 : 'Edible Mushroom', 1 : 'Poisonous Mushroom'}

model = load_model('Model.h5')

model.make_predict_function()


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i)/255.0
    i = cv2.GaussianBlur(i, (5,5),0)
    i = cv2.normalize(i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    return dic[np.argmax(p[0])]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__=='__main__':
	#app.debug = True
	app.run(debug = True)
