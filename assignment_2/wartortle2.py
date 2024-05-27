import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

#Data imnport
data_cal = pd.read_csv(r"C:\Users\Rafał\OneDrive\Pulpit\PRIMO 4\Sem badawcze 3\data_blue\train_blue.csv")
data_val = pd.read_csv(r"C:\Users\Rafał\OneDrive\Pulpit\PRIMO 4\Sem badawcze 3\data_blue\test_blue.csv")


def calc_quantile(data, alpha):
    qhat = []
    # Conditional quantile of class 0
    data0 = data[data['TARGET'] == 0]
    n0 = data0.shape[0]
    q0_level = min(np.ceil((n0 + 1) * (1 - alpha)) / n0, 1)
    qhat.append(np.quantile(data0['CON_SCORE'], q0_level, method='higher'))

    # Conditional quantile of class 1
    data1 = data[data['TARGET'] == 1]
    n1 = data1.shape[0]
    q1_level = min(np.ceil((n1 + 1) * (1 - alpha)) / n1, 1)
    qhat.append(np.quantile(data1['CON_SCORE'], q1_level, method='higher'))

    return qhat

def bin_class_sets(data_cal, data_val, alpha):
    # Calculation of scores for calibration data
    data_cal['CON_SCORE'] = np.where(data_cal['TARGET'] == 1, 1 - data_cal['PRED'], data_cal['PRED'])

    #Quantile
    qhat = calc_quantile(data_cal, alpha)

    # Prediction sets
    data_val['1_IN_PRED_SET'] = data_val['PRED'].apply(lambda x: True if x >= 1 - qhat[1] else False)
    data_val['0_IN_PRED_SET'] = data_val['PRED'].apply(lambda x: True if x <= qhat[0] else False)

    return data_val

def calc_quantile_no_class(data_cal, alpha):
    n = data_cal.shape[0]

    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1)
    qhat = np.quantile(np.abs(data_cal['PRED'] - data_cal['TARGET'])/uncertainty_function_2(data_cal, data_cal), q_level, method='higher')

    return qhat

def uncertainty_interval(data_val, data_cal, q):
    result = uncertainty_function_2(data_val, data_cal)
    interval = np.array([data_val['PRED'] - q * result, data_val['PRED'] + q * result])
    return interval

def draw_plot(data_val, data_cal, q):
    x = data_val.sort_values(by='PRED')
    interval = uncertainty_interval(x, data_cal, q)

    plt.plot(x['PRED'], x['PRED'], label = 'Prediction')
    plt.fill_between(x['PRED'], interval[0,:], interval[1,:], color='r', alpha=.25, label = f'Prediction interval {np.round(1-alpha,2)} level')
    plt.plot(x['PRED'], interval[0,:], label = 'Upper bound prediction', color = 'r', linestyle = '--')
    plt.plot(x['PRED'], interval[1,:], label = 'Lower bound prediction', color = 'r', linestyle = '--')
    plt.legend(loc = 'upper right')
    plt.xlabel('Scores')
    plt.ylabel('Scores')
    plt.title('Conformal regression intervals for prediction scores on test set')
    plt.show()



def uncertainty_function_2(data_val, data_cal):
    model = LogisticRegression()
    model.fit((data_cal['PRED'].values.reshape(-1, 1)), data_cal['TARGET'])
    predict = model.predict_proba(data_val['PRED'].values.reshape(-1, 1))

    return predict[:, 1]




#Params
alpha = 0.1

#Visualisation
predicted_data = bin_class_sets(data_cal, data_val, alpha)
q = calc_quantile_no_class(predicted_data, alpha)
draw_plot(predicted_data, data_cal, q)

