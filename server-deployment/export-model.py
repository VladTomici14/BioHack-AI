import tensorflowjs
from keras.models import load_model

model = load_model("~/Desktop/vladt/codegrouptm/BioHack-AI/birds/dataset/BIRDS-450-(200 X 200)-99.28.h5")
tensorflowjs.converters.save_keras_model(model, "models/")