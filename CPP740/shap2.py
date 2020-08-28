import shap
import numpy as np
import pandas as pd
from  CPP740 import dataPro2 as data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import model_from_json
from keras.models import model_from_yaml
# model = open('model_architecture_1.json', 'r')
yaml_string = open('model_arthitecture_2.yaml', 'r')
model = model_from_yaml(yaml_string)
ac_p,label,ac=data.deal()
aac=data.fe()
ctd=data.CTD()
gaac=data.gaac()
# model=test.do_model()
# model = load_model('my_model.h5')
X=np.concatenate((aac,gaac,ac_p),axis=1)
X=np.array(ac_p)
label=np.array(label)
# x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
# x_train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
# x_test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
# y_train = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
# y_test = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])
x_train = (x_train.reshape((x_train.shape[0], x_train.shape[1], 1)).astype(np.float32))*1000000
x_test = (x_test.reshape((x_test.shape[0], x_test.shape[1], 1)).astype(np.float32))*1000000
y_train = (y_train.reshape((y_train.shape[0], 1)).astype(np.float32))*1000000
y_test = (y_test.reshape((y_test.shape[0], 1)).astype(np.float32))*1000000

# x_train = (np.array(x_train).astype(np.float32))
# x_test = (np.array(x_test).astype(np.float32))
# y_train = (np.array(y_train).astype(np.float32))
# y_test = (np.array(y_test).astype(np.float32))

# select a set of background examples to take an expectation over
# print('==',x_train.shape[0])
# background = x_train[np.random.choice(x_train.shape[0], 320, replace=False)]
# print("background:",background)
# explain predictions of the model on four images
# e = shap.DeepExplainer(model, x_train[1:4])
e = shap.DeepExplainer(model, x_train.astype(np.float64))
print(x_train[1:4])
print("e",e)
# print(x_test[1:2].shape)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
#.astype(np.float32)).astype(np.int)
shap_values = e.shap_values(x_test.astype(np.float64))
shap_values=np.array(shap_values)
shap_values=shap_values.reshape(148,12)*10
print('shap_values',shap_values)
plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['figure.dpi'] = 300
plt.savefig('路径', dpi=300)
shap.summary_plot(shap_values,x_test.reshape(148,12)*10,plot_type='bar',max_display=20)
# plot the feature attributions
# shap.image_plot(shap_values, x_test.reshape(80,12))
