from  CPP740 import dataPro2 as data
from sklearn.svm import SVC
import numpy as np
from keras.models import Sequential
from keras.regularizers import l2
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
import pickle
import scipy.io as sio
import numpy as np
from numpy import linalg as la
import argparse
from keras import backend as K
from sklearn.model_selection import cross_val_score
import  joblib
ac_p,label =data.deal()
aac=data.fe()
ctd=data.CTD()
gaac=data.gaac()
X=np.concatenate((aac,gaac,ac_p),axis=1)
print(X.shape)
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
batch_size =32
epochs = 20
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
label=np.array(label)
num_cross_val=5
all_performance_lstm = []
print("___",len(label[label==1]),len(label[label==0]))
superpa = []
# from sklearn.model_selection import GridSearchCV
# rfc = RandomForestClassifier(n_estimators=79
#                              ,random_state=00
#                              ,criterion='gini'
#                             )
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(X,label)
# print(GS.best_score_,GS.best_params_)
all_labels = []
all_prob_rfc = {}
all_prob_rfc[0] = []
all_prob_svc={}

all_prob_svc[0]=[]
all_prob_lstm={}
all_prob_lstm[0]=[]
all_performance_lstm = []
all_performance_svm = []
all_performance_RF = []
from sklearn.ensemble import RandomForestClassifier
for fold in range(num_cross_val):
    # x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
    # x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
    x_train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
    x_test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
    y_train = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
    y_test = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])
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
    all_labels = all_labels + real_labels
    svc = SVC(kernel="rbf")
    svc = svc.fit(x_train, y_train)
    # rfc.score(x_test,y_test)
    svc_proba = svc.predict(x_test)
    all_prob_svc[0] = all_prob_svc[0] + [val for val in svc_proba]
    y_pred_svc = transfer_label_from_prob(svc_proba)
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(y_test), y_pred_svc, y_test)
    # print(acc)
    print('svm--------',acc, precision, sensitivity, specificity, MCC)
    all_performance_svm.append([acc, precision, sensitivity, specificity, MCC])

    rfc = RandomForestClassifier(n_estimators=29)
    rfc = rfc.fit(x_train, y_train)
    # rfc.score(x_test,y_test)
    rfc_proba = rfc.predict(x_test)
    print(rfc_proba.shape)
    all_prob_rfc[0] = all_prob_rfc[0] + [val for val in rfc_proba]
    y_pred_rfc = transfer_label_from_prob(rfc_proba)
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(y_test), y_pred_rfc, y_test)
    # print(acc)
    print('rf--------', acc, precision, sensitivity, specificity, MCC)
    all_performance_RF.append([acc, precision, sensitivity, specificity, MCC])

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(37, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    opt = optimizers.Adam(lr=0.001)  # 0.01
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    lstm_proba = model.predict_proba(x_test)
    all_prob_lstm[0] = all_prob_lstm[0] + [val for val in lstm_proba]
    y_pred_lstm = transfer_label_from_prob(lstm_proba)
    # # acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(y_test), y_pred_lstm, y_test)
    print(acc, precision, sensitivity, specificity, MCC)
    all_performance_lstm.append([acc, precision, sensitivity, specificity, MCC])

# rfc = rfc.fit(x_train, y_train)
joblib.dump(svc,"svm2.pkl")

# joblib.dump(rfc,"rfc2.pkl")

yaml_string = model.to_yaml()
open('lstm2.yaml', 'w').write(yaml_string)
# # rfc.score(x_test,y_test)
# lstm_proba = rfc.predict(x_test)
# x_train, x_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=10)
# rfc = RandomForestClassifier(n_estimators=29)
# rfc = rfc.fit(x_train, y_train)
# # rfc.score(x_test,y_test)
# lstm_proba = rfc.predict(x_test)
# print(lstm_proba.shape)
# y_pred_xgb = transfer_label_from_prob(lstm_proba)
# acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
# acc, precision, sensitivity, specificity, MCC = calculate_performace(len(y_test), y_pred_xgb, y_test)
# print(acc)
# print(acc, precision, sensitivity, specificity, MCC)
# for i in range(100):
#     rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1,random_state=0)#
#     rfc_s = cross_val_score(rfc,X,label,cv=10).mean()
#     superpa.append(rfc_s)
# print(max(superpa),superpa.index(max(superpa)))
# plt.figure(figsize=[20,5])
# plt.plot(range(1,101),superpa)
# plt.show()


# print('mean performance of QSP')
# print(np.mean(np.array(all_performance_lstm), axis=0))
# print('---' * 50)
print('mean performance of QSP')
print(all_performance_svm)
print('svm_mean-------',np.mean(np.array(all_performance_svm), axis=0))
print('mean performance of QSP')
print(all_performance_RF)
print('rf_mean-------',np.mean(np.array(all_performance_RF), axis=0))
print('mean performance of QSP')
print(all_performance_lstm)
print('lstm_mean-------',np.mean(np.array(all_performance_lstm), axis=0))

plot_roc_curve(all_labels, all_prob_svc[0], 'SVM method')
plot_roc_curve(all_labels, all_prob_rfc[0], 'RF method')
plot_roc_curve(all_labels, all_prob_lstm[0], 'LSTM method')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
# plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
plt.show()

