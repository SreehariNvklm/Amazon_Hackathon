import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense # type: ignore
from keras.models import Sequential # type: ignore
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

myfile = genai.upload_file("img_sample.jpg")
print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", """Can you tell me about the measurements specified above in the format given below with unit name fully specified,
     {measure} {unit}
     ?"""]
)
print(result.text)