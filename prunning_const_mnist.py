import tensorflow as tf
from tensorflow import keras
import tempfile
import tensorflow_model_optimization
import json
import numpy as np
from sklearn.metrics import accuracy_score

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

list_of_results_max = []
list_of_results_avg = []
mnist_dataset = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
x_train = x_train/255
x_test = x_test/255

mnist_sequential_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

mnist_sequential_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mnist_sequential_model.fit(x_train, y_train, epochs=3)

mnist_sequential_model.save("./original_mnist_sequential_model")
_, unmodified_model = tempfile.mkstemp('.h5')
tf.keras.models.save_model(mnist_sequential_model, unmodified_model, include_optimizer=False)
print('Saved temp unmodified model to:', unmodified_model)

_,unmodified_accuracy = mnist_sequential_model.evaluate(x_test, y_test)
print('Accuracy of unmodified model: ', unmodified_accuracy)

converter = tf.lite.TFLiteConverter.from_keras_model(mnist_sequential_model)
unmodified_model_tflite = converter.convert()
_, unmodified_model_tflite_file = tempfile.mkstemp('.tflite')

with open(unmodified_model_tflite_file, 'wb') as f:
  f.write(unmodified_model_tflite)


interpreter = tf.lite.Interpreter(model_content=unmodified_model_tflite)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
interpreter.resize_tensor_input(input_details[0]['index'], (10000, 28, 28))
interpreter.resize_tensor_input(output_details[0]['index'], (10000,))
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
base_tflite_x_test = np.array(x_test, dtype=np.float32)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], base_tflite_x_test)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
prediction_classes = np.argmax(tflite_model_predictions, axis=1)
unmodified_tflite_accuracy_score = accuracy_score(prediction_classes, y_test)





for i in range (0, 10):
  results_for_run = {}
  results_for_run['iteration']=i+1
  results_for_run['sparcity']=i*0.1
  model_for_prunning_deafult = tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(mnist_sequential_model, pruning_schedule=tensorflow_model_optimization.sparsity.keras.ConstantSparsity(0.1*i, 0), block_pooling_type='MAX')
  model_for_prunning_deafult.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  # model_for_prunning_deafult.summary()

  # training with callbacks to debug and log summaries

  logdir = tempfile.mkdtemp()

  callbacks = [
    tensorflow_model_optimization.sparsity.keras.UpdatePruningStep(),
    tensorflow_model_optimization.sparsity.keras.PruningSummaries(log_dir=logdir)
  ]

  model_for_prunning_deafult.fit(x_train, y_train, epochs=3, callbacks=callbacks)
  _, model_for_pruning_deafult_accuracy = model_for_prunning_deafult.evaluate(x_test, y_test)

  print('i is '+str(i))
  print('Unmodified test accuracy:', unmodified_accuracy) 
  print('Default Pruned test accuracy:', model_for_pruning_deafult_accuracy)

  model_for_prunning_deafult_for_export = tensorflow_model_optimization.sparsity.keras.strip_pruning(model_for_prunning_deafult)

  model_for_prunning_deafult_for_export.save("./model_for_prunning_max_for_export"+str(i))
  _, pruned_deafult_keras_file = tempfile.mkstemp('.h5')
  tf.keras.models.save_model(model_for_prunning_deafult_for_export, pruned_deafult_keras_file, include_optimizer=False)
  print('Saved pruned default Keras model to:', pruned_deafult_keras_file)


  converter = tf.lite.TFLiteConverter.from_keras_model(model_for_prunning_deafult_for_export)
  pruned_default_tflite_model = converter.convert()

  _, pruned_default_tflite_file = tempfile.mkstemp('.tflite')

  with open(pruned_default_tflite_file, 'wb') as f:
    f.write(pruned_default_tflite_model)

  print('Saved pruned TFLite model to:', pruned_default_tflite_file)

  print('i is '+str(i))
  print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(unmodified_model)))
  print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_deafult_keras_file)))
  print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_default_tflite_file)))

  
  results_for_run['gzipped_keras_model_size'] = get_gzipped_model_size(unmodified_model)
  results_for_run['gzipped_tflite_model_size'] = get_gzipped_model_size(unmodified_model_tflite_file)
  results_for_run['gzipped_prunned_keras_model_size'] = get_gzipped_model_size(pruned_deafult_keras_file)
  results_for_run['gzipped_prunned_tflite_model_size'] = get_gzipped_model_size(pruned_default_tflite_file)
  results_for_run['base_accuracy'] = unmodified_accuracy
  results_for_run['tflite_base_accuracy'] = unmodified_tflite_accuracy_score
  results_for_run['prunned_tflite_accuracy'] = model_for_pruning_deafult_accuracy

  list_of_results_max.append(results_for_run)

print(list_of_results_max)
with open('results_prunning_const_mnist_max_in_json.json','w') as fout:
    json.dump(list_of_results_max, fout)


for i in range (0, 10):
  results_for_run = {}
  results_for_run['iteration']=i+1
  results_for_run['sparcity']=i*0.1
  model_for_prunning_deafult = tensorflow_model_optimization.sparsity.keras.prune_low_magnitude(mnist_sequential_model, pruning_schedule=tensorflow_model_optimization.sparsity.keras.ConstantSparsity(0.1*i, 0), block_pooling_type='AVG')
  model_for_prunning_deafult.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  # model_for_prunning_deafult.summary()

  # training with callbacks to debug and log summaries

  logdir = tempfile.mkdtemp()

  callbacks = [
    tensorflow_model_optimization.sparsity.keras.UpdatePruningStep(),
    tensorflow_model_optimization.sparsity.keras.PruningSummaries(log_dir=logdir)
  ]

  model_for_prunning_deafult.fit(x_train, y_train, epochs=3, callbacks=callbacks)
  _, model_for_pruning_deafult_accuracy = model_for_prunning_deafult.evaluate(x_test, y_test)

  print('i is '+str(i))
  print('Unmodified test accuracy:', unmodified_accuracy) 
  print('Default Pruned test accuracy:', model_for_pruning_deafult_accuracy)

  model_for_prunning_deafult_for_export = tensorflow_model_optimization.sparsity.keras.strip_pruning(model_for_prunning_deafult)

  model_for_prunning_deafult_for_export.save("./model_for_prunning_avg_for_export_"+str(i))
  _, pruned_deafult_keras_file = tempfile.mkstemp('.h5')
  tf.keras.models.save_model(model_for_prunning_deafult_for_export, pruned_deafult_keras_file, include_optimizer=False)
  print('Saved pruned default Keras model to:', pruned_deafult_keras_file)


  converter = tf.lite.TFLiteConverter.from_keras_model(model_for_prunning_deafult_for_export)
  pruned_default_tflite_model = converter.convert()

  _, pruned_default_tflite_file = tempfile.mkstemp('.tflite')

  with open(pruned_default_tflite_file, 'wb') as f:
    f.write(pruned_default_tflite_model)

  print('Saved pruned TFLite model to:', pruned_default_tflite_file)

  print('i is '+str(i))
  print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(unmodified_model)))
  print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_deafult_keras_file)))
  print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_default_tflite_file)))

  
  results_for_run['gzipped_keras_model_size'] = get_gzipped_model_size(unmodified_model)
  results_for_run['gzipped_tflite_model_size'] = get_gzipped_model_size(unmodified_model_tflite_file)
  results_for_run['gzipped_prunned_keras_model_size'] = get_gzipped_model_size(pruned_deafult_keras_file)
  results_for_run['gzipped_prunned_tflite_model_size'] = get_gzipped_model_size(pruned_default_tflite_file)
  results_for_run['base_accuracy'] = unmodified_accuracy
  results_for_run['tflite_base_accuracy'] = unmodified_tflite_accuracy_score
  results_for_run['prunned_tflite_accuracy'] = model_for_pruning_deafult_accuracy

  list_of_results_avg.append(results_for_run)

print(list_of_results_avg)
with open('results_prunning_const_mnist_avg_in_json.json','w') as fout:
    json.dump(list_of_results_avg, fout)