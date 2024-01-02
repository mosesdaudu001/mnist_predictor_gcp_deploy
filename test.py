import requests

# resp = requests.post("https://getpredictions-pz7paatptq-lm.a.run.app", files={'file': open('./eight.png', 'rb')})
resp = requests.post("http://127.0.0.1:5000", files={'file': open('./eight.png', 'rb')})

print(resp.json())

# import io
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image
# from google.cloud import storage
# import tensorflow.lite as tflite

# from flask import Flask, request, jsonify

# storage_client = storage.Client()
# bucket = storage_client.get_bucket('moses-daudu-bucket')
# blob_classifier = bucket.blob('models/mnist-model.tflite')
# blob_classifier.download_to_filename('tmp/mnist-model.tflite')

# def transform_image(pillow_image):
#   data = np.asarray(pillow_image)
#   data = data / 255.0
#   data = data[np.newaxis, ..., np.newaxis]
#   # --> [1, x, y, 1]
#   data = tf.image.resize(data, [28, 28])
#   data = np.array(data)
#   return data


# def predict(x):
#   predictions = model(x)
#   predictions = tf.nn.softmax(predictions)
#   pred0 = predictions[0]
#   label0 = np.argmax(pred0)
#   return label0

# def hello_world(request):
#   img_file = request.files.get('file')
#   if img_file is None or img_file.filename == "":
#     return jsonify({"error": "no file"})
#   try:
#     image_bytes = img_file.read()
#     pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
#     X = transform_image(pillow_img)

#     interpreter = tflite.Interpreter(model_path='tmp/mnist-model.tflite')
#     interpreter.allocate_tensors()

#     # Get the input and output details
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Ensure the input tensor has the correct shape
#     interpreter.resize_tensor_input(input_details[0]['index'], X.shape)
#     interpreter.allocate_tensors()

#     # Set the input tensor
#     interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))

#     # Run inference
#     interpreter.invoke()

#     # Get the output tensor values
#     preds = interpreter.get_tensor(output_details[0]['index'])

#     # Post-process the output values as needed
#     prediction = np.argmax(preds, axis=1)
#     data = {"prediction": int(prediction)}
#     return jsonify(data)
#   except Exception as e:
#     return jsonify({"error": str(e)})

#   return "OK"
  
