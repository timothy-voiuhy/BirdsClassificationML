import tensorflow as tf
import argparse
from PIL import Image
from pathlib import Path
import logging 
import asyncio
import numpy as np 
import random 
import hashlib 
import os
import sys
import pandas as pd

IMAGE_SIZE = (224, 224)
CSV_MODEL_INFO_FILE = "/media/kali/system2/home/program/AI_ML/AI_ML_DEV_PY/BirdsClassification/birds.csv"
MODELPATH = "/media/kali/system2/home/program/AI_ML/AI_ML_DEV_PY/BirdsClassification/tfBirds_bm.keras"

label_id_dict = {}
df = pd.read_csv(CSV_MODEL_INFO_FILE)
labels = sorted(df["labels"].value_counts().keys())
class_ids = sorted(df["class id"].value_counts().keys())
for label, class_id in zip(labels, class_ids):
    label_id_dict[class_id] = label

def preprocess_image(image):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image/255.0
    return image

def predict_single_image_tf(image_path, model_path = None):
    """Predicts the class of a single crack image using a TensorFlow Keras model.

    Args:
        image_path: Path to the image file.
        model_path: Path to the saved Keras model file (.keras).

    Returns:
        The predicted class
    """

    logging.info("Loading model")
    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = MODELPATH
        model = tf.keras.models.load_model(model_path)
    try:
        if model:
            logging.info("Done loading model")
            # Load and preprocess the image
        # if Path(image_path).exists():
        logging.info(f"Processing image: {image_path}")
        image = Image.open(image_path)
        image = preprocess_image(image)
        # Make the prediction

        logging.info("Model is making prediction")
        prediction = model.predict(image)

        predicted_class = float(np.argmax(prediction, axis=1)[0])+0.0
        _certainity = max(prediction)
        logging.info(f"Model prediction: {predicted_class} with certainity: {_certainity}")
        
        label_name = label_id_dict[predicted_class]
        logging.info(f"Model predicted {label_name} in image")
            # return predicted_class
        # else:
        #     logging.warning(f"Image path: {image_path} does not exist")
    except Exception as e:
        logging.error(f"Failed to load model with erro: {e}")

async def handleClient(reader:asyncio.StreamReader, writer:asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    logging.info(f"Connected to : {addr}")
    # read the data sent by the client
    image_name = "received_image"+str(random.randint(0, 10000))
    hashed_image_name = hashlib.md5(image_name.encode()).hexdigest()+".jpg"; # to make sure the names are different
    with open(hashed_image_name, "wb") as image_file:
        while True:
            data = await reader.read(1024)
            if not data or b"END_OF_IMAGE" in data:
                break # end data stream 
            image_file.write(data[:-len(b"END_OF_IMAGE")])
        logging.info("Image recieved from client")
    predicted_class = int(predict_single_image_tf(hashed_image_name))
    writer.write(predicted_class.to_bytes(4, byteorder ="little"))
    await writer.drain()
    os.remove(hashed_image_name)
    writer.close()
    await writer.wait_closed()

async def runServer(address = "127.0.0.1", server_port = 8082):
    try:
        server = await asyncio.start_server(handleClient, address, server_port)
        addr = server.sockets[0].getsockname()
        logging.info(f"Serving Model Server on : {addr}")
        async with server:
            await server.serve_forever()
    except KeyboardInterrupt as e:
        logging.info("Closing server")
        server.close()
        await server.wait_closed()


def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict with a TensorFlow model")
    parser.add_argument("--predict", action="store_true", help="Predict on a single image")
    parser.add_argument("--image_path", type=str, help="Path to the image file for prediction")
    parser.add_argument("--root", type=str, help="Root directory of the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the saved Keras model file (.keras)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--runServer", action="store_true", help="whether to run a server for the model for remote usage")
    return parser.parse_args()


if __name__ == "__main__":
        logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s -%(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s")

        for logger_name in logging.root.manager.loggerDict:
            if logger_name not in ["__main__"]:
                logging.getLogger(logger_name).setLevel(logging.ERROR)
        args = parse_args()
        if args.runServer:
            asyncio.run(runServer())
        if args.predict:
            if args.image_path is not None:
                predict_single_image_tf(args.image_path, args.model_path)
            else:
                print("Image path not provided")
                sys.exit(1)