from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
from imageio import imread
import traceback 

app = FastAPI()

print("Current working directory:", os.getcwd())

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DMT:
    def __init__(self):
        self.pb = 'dmt.pb'
        self.style_dim = 8
        self.load_model()

    def preprocess(self, img):
        return (img / 255. - 0.5) * 2

    def deprocess(self, img):
        return (img + 1) / 2

    def load_image(self, path):
        img = cv2.resize(imread(path), (256, 256))
        img_ = np.expand_dims(self.preprocess(img), 0)
        return img / 255., img_

    def load_model(self):
        with tf.Graph().as_default():
            output_graph_def = tf.compat.v1.GraphDef()
            with open(self.pb, 'rb') as fr:
                output_graph_def.ParseFromString(fr.read())
                tf.import_graph_def(output_graph_def, name='')

            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
            graph = tf.compat.v1.get_default_graph()
            self.X = graph.get_tensor_by_name('X:0')
            self.Y = graph.get_tensor_by_name('Y:0')
            self.S = graph.get_tensor_by_name('S:0')
            self.X_content = graph.get_tensor_by_name('content_encoder/content_code:0')
            self.X_style = graph.get_tensor_by_name('style_encoder/style_code:0')
            self.Xs = graph.get_tensor_by_name('decoder_1/g:0')
            self.Xf = graph.get_tensor_by_name('decoder_2/g:0')

    def pairwise(self, path_A, path_B):
        A_img, A_img_ = self.load_image(path_A)
        B_img, B_img_ = self.load_image(path_B)
        Xs_ = self.sess.run(self.Xs, feed_dict={self.X: A_img_, self.Y: B_img_})
        result = self.deprocess(Xs_)[0]
        result = (result * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

neural_network = DMT()

@app.post("/process_images")
async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(await image1.read())
            image1_path = f1.name

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(await image2.read())
            image2_path = f2.name

        print(f"Processing images: {image1_path}, {image2_path}")
        result_image = neural_network.pairwise(image1_path, image2_path)
        print("Image processing completed")

        _, buffer = cv2.imencode('.jpg', result_image)
        content = buffer.tobytes()

        os.unlink(image1_path)
        os.unlink(image2_path)

        return Response(content, media_type="image/jpeg")
    except Exception as e:
        print("ERROR OCCURRED")
        traceback.print_exc()
        return Response(content=str(e), status_code=500)


