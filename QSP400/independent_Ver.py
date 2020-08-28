from QSP400 import dataPro as data
import numpy as np
from keras import optimizers
from keras.models import model_from_yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
aac,label=data.fe()
gaac=data.gaac()
ac_p,label=data.deal()
yaml_string = open('model_arthitecture_2.yaml', 'r')
model = model_from_yaml(yaml_string)
yaml_string_lstm = open('lstm.yaml', 'r')
model_lstm = model_from_yaml(yaml_string_lstm)
x_test = np.concatenate(( aac,gaac,ac_p),axis=1)
print(x_test.shape)


all_labels=[]
all_prob = {}
all_prob[0] = []
real_labels = []
for val in label:
    if val == 1:
        real_labels.append(1)
    else:
        real_labels.append(0)
train_label_new = []
# global all_labels
# global all_prob
all_labels = all_labels + real_labels

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.6 else 0 for val in proba]
    return label
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
import joblib
all_prob_svc={}
all_prob_svc[0]=[]
def threeDependent(model):
    svc_proba = model.predict(x_test)
    all_prob_svc[0] = all_prob_svc[0] + [val for val in svc_proba]
    y_pred_svc = transfer_label_from_prob(svc_proba)
    # acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(label), y_pred_svc, label)
    fpr_1, tpr_1, thresholds_1 = roc_curve(all_labels, all_prob_svc[0])  # probas_[:, 1])
    AUC = auc(fpr_1, tpr_1)
    return acc, precision, sensitivity, specificity, MCC,AUC
all_prob_lstm={}
all_prob_lstm[0]=[]
opt = optimizers.Adam(lr=0.001)

def lstmDependent(model):
    opt = optimizers.Adam(lr=0.001)  # 0.01
    model_lstm.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(x_test, label, batch_size=5, epochs=30)
    lstm_proba = model_lstm.predict_proba(x_test)
    print(lstm_proba)
    all_prob_lstm[0] = all_prob_lstm[0] + [val for val in lstm_proba]
    y_pred_lstm = transfer_label_from_prob(lstm_proba)
    #    acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(label), y_pred_lstm, label)
    fpr_4, tpr_4, thresholds_4 = roc_curve(all_labels, all_prob_lstm[0])  # probas_[:, 1])
    AUC = auc(fpr_4, tpr_4)
    return acc, precision, sensitivity, specificity, MCC, AUC

def myDependent(model):
    opt = optimizers.Adam(lr=0.001)  # 0.01
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_test, label, batch_size=5, epochs=30)
    lstm_cnn_proba = model.predict_proba(x_test)
    print(lstm_cnn_proba)
    all_prob[0] = all_prob[0] + [val for val in lstm_cnn_proba]
    y_pred_lstm_cnn = transfer_label_from_prob(lstm_cnn_proba)
    # acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
    print("label",label.shape)
    print(len(y_pred_lstm_cnn))
    acc_lstm_cnn, precision, sensitivity, specificity, MCC = calculate_performace(len(label), y_pred_lstm_cnn, label)
    fpr_3, tpr_3, thresholds_3 = roc_curve(all_labels, all_prob[0])  # probas_[:, 1])
    AUC = auc(fpr_3, tpr_3)
    return acc_lstm_cnn, precision, sensitivity, specificity,MCC,AUC

print("acc, precision, sensitivity, specificity,MCC,AUC")
print("------------------------------------------------")
svm_model = joblib.load("svm.pkl")
acc1, precision1, sensitivity1, specificity1, MCC1, AUC1=threeDependent(svm_model)

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
label = np.array(label)
label = label.reshape((label.shape[0], 1))


yaml_string_lstm = open('lstm.yaml', 'r')
model_lstm = model_from_yaml(yaml_string_lstm)
acc2, precision2, sensitivity2, specificity2, MCC2, AUC2=lstmDependent(model_lstm)

yaml_string = open('model_arthitecture_2.yaml', 'r')
model = model_from_yaml(yaml_string)
acc3, precision3, sensitivity3, specificity3, MCC3, AUC3=myDependent(model)


print("svm",acc1, precision1, sensitivity1, specificity1, MCC1, AUC1)
print('lstm',acc2, precision2, sensitivity2, specificity2, MCC2, AUC2)
print('our',acc3, precision3, sensitivity3, specificity3, MCC3, AUC3)
print("------------------------------------------------")


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)

plt.savefig('路径', dpi=300) #指定分辨


# plt.boxplot(x = (Acc_svm,Acc_rfc,Acc_lstm_cnn,Acc_lstm),
#            patch_artist=True,
#            labels = ('SVM','RF','proposed method','LSTM'), # 添加x轴的刻度标签
#            showmeans=True,showfliers=True,
           # boxprops = {'color':'black','facecolor':'steelblue'},
           # flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':5},
           # meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4},
           #  medianprops={'linestyle':'--','color':'orange'}
           # )
# plot_roc_curve(all_labels, all_prob_svc[0], 'SVM method')
# plot_roc_curve(all_labels, all_prob_rfc[0], 'RF method')
# plot_roc_curve(all_labels, all_prob[0], 'proposed method')
# plot_roc_curve(all_labels, all_prob_lstm[0], 'LSTM method')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([-0.05, 1])
# plt.ylim([0, 1.05])
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
# plt.show()


#         rfc_model = joblib.load("rfc.pkl")
#         rfc_proba = rfc_model.predict(x_test)
#         all_prob_rfc[0] = all_prob_rfc[0] + [val for val in rfc_proba]
#         y_pred_rfc = transfer_label_from_prob(rfc_proba)
# # acc=calculate_performace(len(real_labels), y_pred_xgb, real_labels)
#         acc_rfc, precision, sensitivity, specificity, MCC = calculate_performace(len(label), y_pred_rfc, label)
#         ACC_rfc.append(acc_rfc)