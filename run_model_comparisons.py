import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from numpy import percentile
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as AUC
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.metrics import brier_score_loss, f1_score,make_scorer,matthews_corrcoef,precision_score,precision_recall_fscore_support
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV,Perceptron,SGDClassifier
from sklearn.preprocessing import StandardScaler
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC , LinearSVC

from sklearn.kernel_approximation import Nystroem
import xgboost
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from  tensorflow.keras.utils import to_categorical
from  tensorflow.keras.layers import Conv1D, MaxPooling1D
from  tensorflow.keras.models import Sequential
from  tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from  tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from sklearn.utils import class_weight
from  tensorflow.keras.preprocessing.text import Tokenizer
from  tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
)
import xgboost 


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from  tensorflow.keras.utils import to_categorical
from  tensorflow.keras.layers import Conv1D, MaxPooling1D
from  tensorflow.keras.models import Sequential
from  tensorflow.keras import layers
from  tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from sklearn.utils import class_weight
from  tensorflow.keras.preprocessing.text import Tokenizer
from  tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
)
import shap


def plot_evaluation_results(y,pred,prob):
    summary_plot = plt.figure(figsize=[15, 4])
    cm = confusion_matrix(y_true=y, y_pred=pred)
    plt.subplot(121)
    ax = sns.heatmap(
        cm, annot=True, cmap="Blues", cbar=False, annot_kws={"size": 10}, fmt="g"
    )
    cm_labels = [
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
    ]
    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cm_labels[i])
    plt.title("Confusion Matrix", size=15)
    plt.xlabel("Predicted Values", size=13)
    plt.ylabel("True Values", size=13)

    # Calculating true positive and false positive rates
    fp_rates, tp_rates, _ = roc_curve(y_true=y, y_score=prob)
    roc_auc = AUC(x=fp_rates, y=tp_rates)
    plt.subplot(122)
    plt.plot(
        fp_rates,
        tp_rates,
        color="green",
        lw=1,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")

    # Creating current decision point plot
    tn, fp, fn, tp = [i for i in cm.ravel()]
    plt.plot(fp / (fp + tn), tp / (tp + fn), "bo", markersize=7, label="Decision Point")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", size=11)
    plt.ylabel("True Positive Rate", size=11)
    plt.title("ROC Curve", size=10)
    plt.legend(loc="lower right")
    
def recall_m(y_true, y_pred):
        true_positives = np.dot(y_true,y_pred)
        possible_positives = np.sum(y_true)
        recall = true_positives / float(possible_positives)
        return recall

def precision_m(y_true, y_pred):
        true_positives = np.dot(y_true,y_pred)
        predicted_positives = np.sum(y_pred)
        precision = true_positives / float(predicted_positives)
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def compare_cohorts(data_path, test_data_path, def_path, results_path, nruns, gt_col = 'PU_mention_in_both'):
    cohorts = ["ours", "Jamia","egem", "amia"]
    seeds = np.array(random.sample(range(100), nruns))
    performance_parameters = {}
    for cohort in cohorts:
        model_parameters = {}
        gt_auprcs = []
        gt_f1s = []
        gt_precisions = []
        gt_recalls = []
        gt_aurocs = []
        gt_accuracies = []

        gt_auprcs_lr = []
        gt_f1s_lr = []
        gt_precisions_lr = []
        gt_recalls_lr = []
        gt_aurocs_lr = []
        gt_accuracies_lr = []

        gt_auprcs_nn = []
        gt_f1s_nn = []
        gt_precisions_nn = []
        gt_recalls_nn = []
        gt_aurocs_nn = []
        gt_accuracies_nn = []

        for split_idx in range(1,11):
            i = 0
            test_data = pd.read_csv( test_data_path + "combined_exp_test_note_"+ str(split_idx) + ".csv")
            test_x = test_data.TEXT
            test_y = test_data.ground_truth

            print (test_data.shape)
            test_id = test_data['HADM_ID']
            
            cohort_data = def_path + "limnote_only_note_threepapers_" + cohort + ".csv"

            data = pd.read_csv(cohort_data)
            train_data = data[~data['HADM_ID'].isin(test_id.values)]
            train_x = train_data.TEXT
            train_y = train_data['PU_mention_in_both']

            vect2 = TfidfVectorizer(max_features=5000, min_df=0.01, max_df=0.85, stop_words = 'english')
            vect2.fit(train_x)
            feat_names = vect2.get_feature_names()

            tokenizer = Tokenizer(num_words=5000)
            custom_word_index = {w: i for i, w in enumerate(feat_names)}
            tokenizer.word_index = custom_word_index
            maxlen = 800
            vocab_size = len(tokenizer.word_index) + 1

            train_x_nn = tokenizer.texts_to_sequences(train_x)
            test_x_nn = tokenizer.texts_to_sequences(test_x)

            train_x_nn = pad_sequences(train_x_nn, padding="post", maxlen=maxlen, truncating="pre")
            test_x_nn = pad_sequences(test_x_nn, padding="post", maxlen=maxlen, truncating="pre")
            
            test_x = vect2.transform(test_x)
            train_x = vect2.transform(train_x)
            for seed in seeds:
                i += 1

                # neural network

                # Model architecture
                embedding_dim = 100

                def create_model(vocab_size = vocab_size, embedding_dim = embedding_dim , maxlen = maxlen):
                    mdl = Sequential()
                    mdl.add(
                        layers.Embedding(
                            input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen
                        )
                    )
                    mdl.add(layers.GlobalMaxPool1D())
                    # mdl.add(LSTM(10))
                    mdl.add(layers.Dense(16, activation="relu"))
                    mdl.add(Dropout(0.1))
                    mdl.add(layers.Dense(1, activation="sigmoid"))
                    mdl.compile(optimizer="adam", loss="binary_crossentropy", metrics=['Precision'])

                    return mdl


                modelll = KerasClassifier(build_fn=create_model)


                # Fit the model
                class_weights = class_weight.compute_class_weight(
                    "balanced", np.unique(train_y), train_y
                )
                class_weight_dict = dict(enumerate(class_weights))


                batch_size_lists = [64, 128, 256]
    #             batch_size_lists = [ 128]

                nn_param_grid = dict(batch_size = batch_size_lists)

                cv_nn = GridSearchCV(scoring ='average_precision',estimator=modelll, param_grid=nn_param_grid, cv= 5,verbose=2, n_jobs=-1)
                cv_nn.fit(train_x_nn,train_y ,epochs=10, class_weight=class_weight_dict)
                gridmod_nn = cv_nn.best_estimator_

                pred_test_y_nn  = gridmod_nn.predict(test_x_nn)
                pred_train_y_nn  = gridmod_nn.predict(train_x_nn)
                pred_test_prob_nn  = gridmod_nn.predict_proba(test_x_nn)[:,1]
                pred_train_prob_nn  = gridmod_nn.predict_proba(train_x_nn)[:,1]

                pred_test_y_nn = np.array([m[0] for m in pred_test_y_nn])

                auprc_nn  = average_precision_score(test_y, pred_test_prob_nn )
                auc_nn  = roc_auc_score(test_y, pred_test_prob_nn )
                recall_nn  = recall_m(test_y, pred_test_y_nn )
                precision_nn  = precision_m(test_y, pred_test_y_nn )
                f1_nn  = f1_m(test_y, pred_test_y_nn )
                print (pred_test_y_nn)
                print (pred_test_prob_nn)
                accuracy_nn  = np.sum(test_y == pred_test_y_nn )/float(test_y.shape[0])
                gt_auprcs_nn .append(auprc_nn)
                gt_f1s_nn .append(f1_nn )
                gt_precisions_nn.append(precision_nn)
                gt_recalls_nn.append(recall_nn )
                gt_aurocs_nn .append(auc_nn )
                gt_accuracies_nn .append(accuracy_nn)
                if i == 1:
                    first_pos_idx = list(test_y.index).index(test_y[test_y==1].index[0])
                    explainer = shap.KernelExplainer(gridmod_nn.predict_proba, train_x_nn[:10,:])
                    shap_values = explainer.shap_values(test_x_nn[first_pos_idx,:])
                    import_dict = {}
                    for l in range(shap_values[1].shape[0]):
                        value = shap_values[1][l]
                        if value != 0:
    #                         print(test_x_nn[0,l])
                            key = feat_names[test_x_nn[first_pos_idx,l]]
                            if key not in import_dict:
                                import_dict[key] = value
                            else:
                                if abs(import_dict[key]) < abs(value):
                                    import_dict[key] = value
                    feature_importance = pd.DataFrame(columns=["feature", "importance"])
                    feature_importance["importance"] = import_dict.values()
                    feature_importance["feature"] = import_dict.keys()
                    # feature_importance["importance"] = pd.to_numeric(feature_importance["importance"])
                    fi_keywords = feature_importance.sort_values(by="importance",key= lambda x: abs(x), ascending=False)
                    fi_keywords.set_index("feature", inplace=True, drop=True)
                    fi_keywords.reset_index(inplace=True)


                    fi_keywords.sort_values(
                        by="importance", inplace=True, key= lambda x: abs(x), ascending=False
                    )


                    fi_keywords.to_csv(
                        results_path 
                        + str((i) * 100)                
                        + cohort
                        + "_combiinn_words.csv",
                        index=False,
                    )

                
                lrc = HistGradientBoostingClassifier(random_state = 5)

                lr_param_grid = {'max_iter' : [400,800],
    #                            'min_samples_leaf':[1,3,5],
                               # 'max_depth': [5,10,20,40],
    #                            'min_samples_split': [2,4,8],
                }
                cv_lr = GridSearchCV(scoring ='average_precision',estimator=lrc, param_grid=lr_param_grid, cv= 5,verbose=2)
                cv_lr.fit(train_x.toarray(),train_y)
                gridmod_lr = cv_lr.best_estimator_
                pred_test_y_lr = gridmod_lr.predict(test_x.toarray())
                pred_train_y_lr = gridmod_lr.predict(train_x.toarray())
                pred_test_prob_lr = gridmod_lr.predict_proba(test_x.toarray())[:,1]
                pred_train_prob_lr = gridmod_lr.predict_proba(train_x.toarray())[:,1]
                auprc_lr = average_precision_score(test_y, pred_test_prob_lr)
                auc_lr = roc_auc_score(test_y, pred_test_prob_lr)
                recall_lr = recall_m(test_y, pred_test_y_lr)
                precision_lr = precision_m(test_y, pred_test_y_lr)
                f1_lr = f1_m(test_y, pred_test_y_lr)
                accuracy_lr = np.sum(test_y == pred_test_y_lr)/float(test_y.shape[0])
                gt_auprcs_lr.append(auprc_lr)
                gt_f1s_lr.append(f1_lr)
                gt_precisions_lr.append(precision_lr)
                gt_recalls_lr.append(recall_lr)
                gt_aurocs_lr.append(auc_lr)
                gt_accuracies_lr.append(accuracy_lr)

                print ("done one seed at")
                print (seed)
                print (seeds)
                print (cohort)
                
                preds_data = np.column_stack((test_id ,test_y,pred_test_y_nn,pred_test_y_lr))
                preds_df = pd.DataFrame(data = preds_data, columns = ['HADM_ID','ground_label','neural_label','gb_label'])
                preds_df.to_csv(
                        results_path
                        + str((i) * 100)                
                        + cohort
                        + "_combiipreds.csv",
                        index=False,
                    )
            gt_avg_auprc_lr = np.mean(np.array(gt_auprcs_lr))
            model_parameters["GB_auprcs"] = list(gt_auprcs_lr)
            model_parameters["GB_auroc"] = list(gt_aurocs_lr)
            model_parameters["GB_precision"] = list(gt_precisions_lr)
            model_parameters["GB_recall"] = list(gt_recalls_lr)
            model_parameters["GB_f1s"] = list(gt_f1s_lr)
            model_parameters["GB_accuracies"] = list(gt_accuracies_lr)
            model_parameters["GB_avg_auprc"] = gt_avg_auprc_lr

            gt_avg_auprc_nn = np.mean(np.array(gt_auprcs_nn))
            model_parameters["NN_auprcs"] = list(gt_auprcs_nn)
            model_parameters["NN_auroc"] = list(gt_aurocs_nn)
            model_parameters["NN_precision"] = list(gt_precisions_nn)
            model_parameters["NN_recall"] = list(gt_recalls_nn)
            model_parameters["NN_f1s"] = list(gt_f1s_nn)
            model_parameters["NN_accuracies"] = list(gt_accuracies_nn)
            model_parameters["NN_avg_auprc"] = gt_avg_auprc_nn

            print ("finished")
            print (cohort)
            print (i)

            performance_parameters[cohort] = model_parameters
            randint = np.random.randint(100)
            additiol_file_name = (
            str(randint) +  (cohort) 
            + str(split_idx) + ""
            )
            with open(
            data_path + additiol_file_name + "full_experiment_fullcomb.json", "w"
            ) as fp:
                json.dump(performance_parameters, fp)


                
test_data_path = '/data/mimic/PUI_data/data/agreed/'
def_path = "/data/mimic/PUI_data/data/"
results_path = "/home/mani/nursing/PU_nursing/jamia_results/"
nruns = 1

compare_cohorts('/home/mani/nursing/PU_nursing/', test_data_path, def_path, results_path, nruns,  'ground_truth')

