from QSP400 import dataPro as data
from keras.models import Sequential
from keras.regularizers import l2
import keras.backend as K
from keras.layers import LSTM, Dense, Dropout, Activation, initializers, GRU, SimpleRNN,ConvLSTM2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import optimizers
import matplotlib.pyplot as plt
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,BatchNormalization, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Embedding,Bidirectional,LeakyReLU
from keras.utils import plot_model
# from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import scipy.io as sio
import numpy as np
from numpy import linalg as la
import argparse
from keras import backend as K
from sklearn.model_selection import cross_val_score
import numpy as np
ac_p,label,ac,=data.deal()

aac=data.fe()
ctd=data.CTD()
gaac=data.gaac()
X=np.concatenate((aac,gaac,ac_p),axis=1)
# X=np.array(ac_p)
# print(X)
def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    print('tp:',tp,'fn:',fn,'tn:',tn,'fp:',fp)
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC
    # return acc


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.6 else 0 for val in proba]
    return label


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)

# define parameters
batch_size =20
epochs = 20
# all_labels=[]
# all_prob = {}
# all_prob[0] = []
label=np.array(label)
num_cross_val=5
all_performance_lstm = []
X_f=[np.concatenate((aac,gaac,ac_p),axis=1),np.concatenate((aac,gaac),axis=1),np.concatenate((aac,ac_p),axis=1),
     np.concatenate((gaac, ac_p), axis=1)]
# X_f=[np.concatenate((aac,gaac,ac_p),axis=1),np.concatenate((aac,gaac),axis=1)]
# print("___",len(label[label==1]),len(label[label==0]))
superpa = []
All_labels=[]
All_prob={}
All_prob[0]=[]
def do_model():
    for X in X_f:
        all_labels=[]
        all_prob = {}
        all_prob[0] = []
        # x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
        x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_train = y_train.reshape((y_train.shape[0], 1))
        y_test = y_test.reshape((y_test.shape[0], 1))
        real_labels = []

        for val in y_test:
            if val == 1:
                real_labels.append(1)
            else:
                real_labels.append(0)

        train_label_new = []
        for val in y_train:
            if val == 1:
                train_label_new.append(1)
            else:
                train_label_new.append(0)
        global    all_labels
        global    all_prob
        all_labels = all_labels + real_labels
        print("**", type(x_train))
        print(x_train.shape, y_train.shape)
        model = Sequential()
        model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', input_shape=(x_train.shape[1], 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(8, kernel_size=3, strides=1, padding='same'))
        model.add(Conv1D(16, kernel_size=3, strides=1, padding='valid'))
        model.add(Dropout(0.2))
        model.add(Conv1D(8, kernel_size=3, strides=1, padding='same'))
        model.add(Conv1D(16, kernel_size=3, strides=1, padding='valid'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))
        model.add(Activation('relu'))
        model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
        model.add(Conv1D(32, kernel_size=3, strides=1, padding='valid'))
        model.add(BatchNormalization())
        model.add(Conv1D(32, kernel_size=3, strides=1, padding='same'))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding='same'))
        # model.add(Conv1D(32, kernel_size=3, strides=1, padding='same'))

        model.add(Activation('relu'))
        # model.add(Flatten())
        model.add(Dropout(0.2))
        # model.add(LSTM(32, return_sequences=True))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        # model.add(Dropout(0.2))
        model.add(Flatten())
        # model.add(Dense(512, kernel_initializer='he_normal', activation='relu', W_regularizer=l2(0.01)))
        # model.add(Dense(128, kernel_initializer='he_normal', activation='relu', W_regularizer=l2(0.01)))
        # model.add(Dense(64, kernel_initializer='he_normal', activation='relu', W_regularizer=l2(0.01)))
        # model.add(Dense(16, kernel_initializer='he_normal', activation='relu', W_regularizer=l2(0.01)))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        opt = optimizers.Adam(0.00035)  # 0.01
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        print("Train...")
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        # json_string = model.to_json()
        # open('model_architecture_1.json', 'w').write(json_string)
        # yaml_string = model.to_yaml()
        # open('model_arthitecture_2.yaml', 'w').write(yaml_string)
        lstm_proba = model.predict_proba(x_test)
        # print(lstm_proba)
        # print(y_test)
        all_prob[0] = all_prob[0] + [val for val in lstm_proba]
        y_pred_xgb = transfer_label_from_prob(lstm_proba)
        # acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(y_test), y_pred_xgb, y_test)
        # print(acc)
        # superpa.append(acc)
        All_labels.append(all_labels)
        All_prob[0].append(all_prob[0])
        print(acc, precision, sensitivity, specificity, MCC)

        all_performance_lstm.append([acc, precision, sensitivity, specificity, MCC])
        print(all_performance_lstm)
        print('---' * 50)
    return  model
model=do_model()
yaml_string = model.to_yaml()
open('model_arthitecture_2.yaml', 'w').write(yaml_string)
    # background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    #
    # # explain predictions of the model on three images
    # e = shap.DeepExplainer(model, background)
    # # ...or pass tensors directly
    # # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    # shap_values = e.shap_values(x_test[1:5])
# print("--",(superpa))
# plt.figure(figsize=[7,5])
# plt.plot(np.arange(0.0005,0.007,0.001),superpa,'b--')
# plt.ylim([0, 1.05])
# plt.xlabel('learning rate')
# plt.ylabel('accuracy')
# plt.show()
# print("11111",len(All_labels[0]))
# print("22222",len(All_prob[0][0]),len(all_prob[0]))
# print("11111",All_labels.shape)
# print("22222",All_prob.shape)
print('mean performance of QSP')
print(np.mean(np.array(all_performance_lstm), axis=0))
print('---' * 50)
plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['figure.dpi'] = 300
plt.savefig('路径', dpi=300)
plt.plot(linestyle='--')
plot_roc_curve(All_labels[0], All_prob[0][0], 'proposed method')
plt.plot(linestyle=',')
plot_roc_curve(All_labels[1], All_prob[0][1], 'AAC+GAAC method')
plt.plot(linestyle='--')
plot_roc_curve(All_labels[2], All_prob[0][2], 'AAC+AC_P method')
plt.plot(linestyle='--')
plot_roc_curve(All_labels[3], All_prob[0][3], 'GAAC+AC_P method')
# plot_roc_curve(All_labels[4], All_prob[0][4], 'gaac+ac_p method','r.')
plt.plot(linestyle='--')

# plt.plot(All_labels[0], All_prob[0][0],label='proposed method',color='g',linewidth=2,linestyle=':')
# plt.plot(All_labels[1], All_prob[0][1],label='aac+gaac method',color='w',linewidth=2,linestyle=':')
# plt.plot(All_labels[2], All_prob[0][2],label='aac+ac_p method',color='r',linewidth=2,linestyle=':')
# plt.plot(All_labels[3], All_prob[0][3],label='gaac+ac_p method',color='b',linewidth=2,linestyle=':')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
# plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
plt.show()



# your code

K.clear_session()

