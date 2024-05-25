import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #type: ignore


#Data imnport
data_cal = pd.read_csv(r"C:\Users\aleks\OneDrive - Politechnika Warszawska\SEM10\Allegro\data_blue\train_blue.csv")
data_val = pd.read_csv(r"C:\Users\aleks\OneDrive - Politechnika Warszawska\SEM10\Allegro\data_blue\test_blue.csv")


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


def draw_histogram(data, q):
    
    # Create a histogram with 15 bins
    plt.hist(np.log(data), bins=15, color='aqua', edgecolor='black')


    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()
    xlim = np.linspace(xmin, xmax, 1000)
    ylim = np.zeros_like(xlim) + ymax
    
    # Add a dashed horizontal line at x=q0 (x = 1 - q1)
    plt.axvline(x=np.log(q[0]), color='orange', linestyle='--', label='q0 = {}'.format(round(np.log(q[0]), 3)))
    plt.fill_between(xlim, ylim, where = (xlim <= np.log(q[0])), color='orange', alpha=0.2)
    plt.axvline(x=np.log(1 - q[1]), color='brown', linestyle='--', label='1 - q1 = {}'.format(round(np.log(1 - q[1]), 3)))
    plt.fill_between(xlim, ylim, where = (xlim >= np.log(1 - q[1])), color='brown', alpha=0.2)
    plt.xlabel('Log Predict')
    plt.ylabel('Counts')
    plt.title('Conformal Scores')
    plt.legend(loc='upper right')

    plt.show()


#Params
alpha = 0.1

#Visualisation


#All data
predicted_data = bin_class_sets(data_cal, data_val, alpha)
q = calc_quantile(data_cal, alpha)
draw_histogram(predicted_data['PRED'], q)


#SEG1 = 1
data_cal1 = data_cal[data_cal['SEG_1'] == '1']
data_val1 = data_val[data_val['SEG_1'] == '1']

predicted_data1 = bin_class_sets(data_cal1, data_val1, alpha)
q = calc_quantile(data_cal1, alpha)
draw_histogram(predicted_data1['PRED'], q)

#SEG1 = 2
data_cal2 = data_cal[data_cal['SEG_1'] == '2']
data_val2 = data_val[data_val['SEG_1'] == '2']

predicted_data2 = bin_class_sets(data_cal2, data_val2, alpha)
q = calc_quantile(data_cal2, alpha)
draw_histogram(predicted_data2['PRED'], q)

#SEG1 = 3
data_cal3 = data_cal[data_cal['SEG_1'] == '3']
data_val3 = data_val[data_val['SEG_1'] == '3']

predicted_data3 = bin_class_sets(data_cal3, data_val3, alpha)
q = calc_quantile(data_cal3, alpha)
draw_histogram(predicted_data3['PRED'], q)

#SEG1 = 4
data_cal4 = data_cal[data_cal['SEG_1'] == '4']
data_val4 = data_val[data_val['SEG_1'] == '4']

predicted_data4 = bin_class_sets(data_cal4, data_val4, alpha)
q = calc_quantile(data_cal4, alpha)
draw_histogram(predicted_data4['PRED'], q)

#SEG1 = 5
data_cal5 = data_cal[data_cal['SEG_1'] == '5']
data_val5 = data_val[data_val['SEG_1'] == '5']

predicted_data5 = bin_class_sets(data_cal5, data_val5, alpha)
q = calc_quantile(data_cal5, alpha)
draw_histogram(predicted_data5['PRED'], q)
