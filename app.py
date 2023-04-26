import gradio as gr
import pickle
import numpy as np

model = pickle.load(open('model_RMSprop.pickle', 'rb'))

def recognize_digit(input):
  input = input.reshape((1,784))
  
  prediction = np.squeeze(model.predict(input))
  label = [0,1,2,3,4,5,6,7,8,9]
  output = dict(zip(label, prediction.tolist()))
  return output
  
gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="label").launch(debug = True)