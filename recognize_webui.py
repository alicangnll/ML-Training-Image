from flask import render_template, request, Flask
import base64, cv2, random, os, string
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)
model_file = "model.h5"
verbose = int(0)

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return str(result_str)

@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    req = str(request.form.get("current_image"))
    # Preparing image...
    f = open("temp.png", "wb")
    f.write(base64.decodebytes(req.encode('utf-8')))
    f.close()
    im = cv2.imread("temp.png")
    h, w, c = im.shape
    img = Image.open("temp.png")
    img.resize((w, h))
    image = np.expand_dims(img, axis=0)
    image = (np.float32(image))
    image /= 255.
    # Recognize Image
    f = tf.lite.Interpreter(model_file)
    f.allocate_tensors()
    i = f.get_input_details()[0]
    if int(verbose) == 1:
        print("=== Input of tflite model: " , i)
    o = f.get_output_details()[0]
    if int(verbose) == 1:
        print("=== Output of tflite model: ", o)
    f.set_tensor(i["index"], image)
    f.invoke()
    y = f.get_tensor(o["index"])
    if int(verbose) == 1:
        print("=== Results from recognizing the image with Tf-lite model: ", y)
    os.remove("temp.png")
    return "\n=== Label of the image that we recognize with Tf-lite model: {:d}\n".format(np.argmax(y))
    
@app.route("/saveimage", methods=["GET"])
def saveimage():
    return render_template("save.html")

@app.route("/save", methods=["POST"])
def save():
    req = str(request.form.get("current_image"))
    cat = str(request.form.get("kategori"))
    rnd = str(get_random_string(8))

    os.makedirs("./tf_files/validation/" + cat)
    os.makedirs("./tf_files/train/" + cat)

    if(os.path.exists("./tf_files/train/" + cat)):
        f = open("./tf_files/train/" + cat + "/rename_" + rnd + ".png", "wb")
        f.write(base64.decodebytes(req.encode('utf-8')))
        f.close()

        f2 = open("./tf_files/validation/" + cat + "/rename_" + rnd + ".png", "wb")
        f2.write(base64.decodebytes(req.encode('utf-8')))
        f2.close()
        return "OK"
    else:
        return "Error : File exist"

app.run(host="0.0.0.0", port=81)