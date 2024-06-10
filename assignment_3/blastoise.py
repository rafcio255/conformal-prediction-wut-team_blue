import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import cluster
from sklearn import mixture
from sklearn.impute import KNNImputer

##############################  Ładowanie danych ####################################

data_cal = pd.read_csv(r"C:\Users\Rafał\OneDrive\Pulpit\PRIMO 4\Sem badawcze 3\data_blue\train_blue.csv")
data_val = pd.read_csv(r"C:\Users\Rafał\OneDrive\Pulpit\PRIMO 4\Sem badawcze 3\data_blue\test_blue.csv")

columns_feat = [f"FEAT_{i+1}" for i in range(11)]
feat_cal = data_cal[columns_feat]

feat_cal_noimput = feat_cal[feat_cal.drop(columns=['FEAT_6']).notnull().all(1)]
feat_cal_noimput = feat_cal_noimput.drop(columns=['FEAT_6'])

##############################  Imputacja ####################################

# Lista kolumn, które nie będą imputowane
exclude_columns = ['PRED', 'PRED_BINNED', 'TARGET', 'SEG_1', 'SEG_2', 'SEG_3']

# Imputacja dla data_cal
data_cal_imput_mean = data_cal.copy()
for col in data_cal.columns:
    if col not in exclude_columns:
        data_cal_imput_mean[col] = data_cal[col].fillna(data_cal[col].mean())

# Imputacja dla data_val
data_val_imput_mean = data_val.copy()
for col in data_val.columns:
    if col not in exclude_columns:
        data_val_imput_mean[col] = data_val[col].fillna(data_val[col].mean())

feat_cal_imput_mean = data_cal_imput_mean[columns_feat]
feat_val_imput_mean = data_val_imput_mean[columns_feat]





##############################  Funkcje ####################################

# Liczenie rzeczywistego prawdopodobieństwa
def uncertainty_knn(data_cal, data_val, k = 1000):

    NN = NearestNeighbors(n_neighbors = k)

    data_cal_x = np.array(data_cal.drop(['PRED', 'PRED_BINNED', 'TARGET', 'SEG_1', 'SEG_2', 'SEG_3', 'Cluster_miss'], axis = 1))
    data_val_x = np.array(data_val.drop(['PRED', 'PRED_BINNED', 'TARGET', 'SEG_1', 'SEG_2', 'SEG_3', 'Cluster_miss', 'Cluster'], axis = 1))
    data_cal_y = np.array(data_cal['TARGET'])
    NN.fit(data_cal_x)

    kneigh = NN.kneighbors(data_val_x, return_distance = False)
    targets = data_cal_y[kneigh]
    count_df = [i.sum() for i in targets]
    new_pred = np.divide(count_df, k)
    
    return new_pred


# Funkcja rysująca macierz korelacji
def plot_correlation_matrix(df, title='Correlation Matrix', cmap='Blues'):
    """
    Rysuje macierz korelacji dla zmiennych w dataframe.
    
    Parametry:
    df (pd.DataFrame): DataFrame zawierający dane
    title (str): Tytuł wykresu
    cmap (str): Koloryzacja wykresu (domyślnie 'YlGnBu')
    """
    # Sprawdzanie, czy są jakiekolwiek braki danych
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values. Please handle missing data before plotting correlation matrix.")
    
    # Obliczanie macierzy korelacji
    correlation_matrix = df.corr()

    # Rysowanie wykresu
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, cbar_kws={'shrink': .8})
    plt.title(title)
    plt.show()


# Wyliczanie q
def calc_quantile_no_class(data_cal, alpha):
    n = data_cal.shape[0]

    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1)
    qhat = np.quantile(np.abs(data_cal['PRED'] - data_cal['TARGET'])/data_cal['u'], q_level, method='higher')

    return qhat


# Wyliczanie przedziału niepewności
def uncertainty_interval(data_cal, data_val, alpha):
    q = calc_quantile_no_class(data_cal, alpha)
    data_val['DOWN'] = np.maximum(data_val['PRED'] - q * data_val['u'], np.full(len(data_val['PRED']), 0))
    data_val['UP'] = np.minimum(data_val['PRED'] + q * data_val['u'], np.full(len(data_val['PRED']), 1))
    return data_val


# Rysowanie wykresu
def draw_plot(data_val):
    x = data_val.sort_values(by='PRED')

    plt.plot(x['PRED'], x['PRED'], label = 'Prediction')
    plt.fill_between(x['PRED'], x['DOWN'], x['UP'], color='r', alpha=.25, label = f'Prediction interval {np.round(1-alpha,2)} level')
    plt.plot(x['PRED'], x['DOWN'], label = 'Upper bound prediction', color = 'r', linestyle = '--')
    plt.plot(x['PRED'], x['UP'], label = 'Lower bound prediction', color = 'r', linestyle = '--')
    plt.legend(loc = 'upper right')
    plt.xlabel('Scores')
    plt.ylabel('Scores')
    plt.title('Conformal regression intervals for prediction scores on test set')
    plt.show()


# Funkcja niepewności zależna od ilości braków danych
def uncertainty_na(data_val):
    data_val['u'] = (data_val['Cluster_miss'] + 1) / 10000
    return data_val


# Funkcja niepewności zależna od odległości od najbliższych sąsiadów
def calculate_nearest_neighbors_between_dfs(data_cal, data_val, n_neigh):
    # Lista kolumn do użycia w obliczeniach najbliższych sąsiadów
    feature_columns = [f'FEAT_{i}' for i in range(1, 12)]
    
    
    # Inicjalizacja modelu Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neigh)
    nbrs.fit(data_cal[feature_columns])
    
    # Wyliczenie odległości i indeksów sąsiadów dla obserwacji z data_val względem data_cal
    distances, indices = nbrs.kneighbors(data_val[feature_columns])
    
    # Dodanie kolumny 'u' z sumą odległości do najbliższych sąsiadów
    data_val['u'] = distances.sum(axis=1) / 100
    
    return data_val


# 
def average_u_column(df1, df2):     
    result_df = df1.copy()
    # Dodanie nowej kolumny 'u' z średnią wartością z kolumn 'u' w df1 i df2
    result_df['u'] = (df1['u'] + df2['u']) / 2
    
    return result_df




################### conditional coverage check ##############################

def feature_stratified_coverage_metric(df):

    # Grupowanie danych po kolumnie 'clust'
    grouped = df.groupby('cluster')
    
    # Inicjalizacja listy na przechowywanie pokrycia dla każdej grupy
    coverage_list = []
    
    # Iteracja po grupach
    for name, group in grouped:
        # Liczenie obserwacji mieszczących się w przedziale ufności
        within_interval = ((group['pred'] >= group['DOWN']) & (group['pred'] <= group['UP'])).sum()
        total = group.shape[0]
        coverage = within_interval / total
        coverage_list.append(coverage)
    
    # Obliczanie minimalnego pokrycia
    min_coverage = min(coverage_list)
    
    return min_coverage


###################################  clustrowanie #####################################

#clustorwanie kMeans
# Definiowanie liczby klastrów
num_clusters = 10
# Tworzenie obiektu KMeans
kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=42)
# Dopasowanie modelu do danych
kmeans.fit(feat_val_imput_mean)

# Przypisanie klastrów do danych
data_val_imput_mean['Cluster'] = kmeans.labels_


#clustrowanie po missingach:

# Dodajemy kolumnę z liczbą braków danych w każdej obserwacji
data_val_imput_mean['Cluster_miss'] = data_val.isna().sum(axis=1)
data_cal_imput_mean['Cluster_miss'] = data_cal.isna().sum(axis=1)

def missing_data_summary(feat):
    # Dodajemy kolumnę z liczbą braków danych w każdej obserwacji
    feat_cal['cluster_miss'] = feat_cal.isna().sum(axis=1)
    
    # Obserwacje bez braków danych
    no_missing = feat_cal[feat_cal['cluster_miss'] == 0].shape[0]
    
    # Obserwacje z 1 brakiem
    one_missing = feat_cal[feat_cal['cluster_miss'] == 1]
    one_missing_summary = one_missing.isna().sum().sort_values(ascending=False)
    one_missing_summary = one_missing_summary[one_missing_summary > 0]
    
    # Obserwacje z 2 brakami
    two_missing = feat_cal[feat_cal['cluster_miss'] == 2]
    two_missing_combinations = {}
    for combo in combinations(feat_cal.columns[:-1], 2):
        combo_count = two_missing[two_missing[list(combo)].isna().sum(axis=1) == 2].shape[0]
        if combo_count > 0:
            two_missing_combinations[combo] = combo_count
    
    # Obserwacje z 3 brakami
    three_missing = feat_cal[feat_cal['cluster_miss'] == 3]
    three_missing_combinations = {}
    for combo in combinations(feat_cal.columns[:-1], 3):
        combo_count = three_missing[three_missing[list(combo)].isna().sum(axis=1) == 3].shape[0]
        if combo_count > 0:
            three_missing_combinations[combo] = combo_count
    
    # Obserwacje z 4 i więcej brakami
    more_than_three_missing = {}
    for i in range(4, feat_cal.shape[1]):
        count = feat_cal[feat_cal['cluster_miss'] == i].shape[0]
        if count > 0:
            more_than_three_missing[f'{i} missing'] = count
    
    # Tworzenie raportu
    summary = {
        'No Missing Data': no_missing,
        'One Missing Data': one_missing_summary.to_dict(),
        'Two Missing Data': two_missing_combinations,
        'Three Missing Data': three_missing_combinations,
        'Four or More Missing Data': more_than_three_missing
    }
    
    return summary



################### conditional coverage check ##############################

def feature_stratified_coverage_metric(df):
    
    # Grupowanie danych po kolumnie 'clust'
    grouped = df.groupby('Cluster')
    
    # Inicjalizacja listy na przechowywanie pokrycia dla każdej grupy
    coverage_list = []
    
    # Iteracja po grupach
    for name, group in grouped:
        # Liczenie obserwacji mieszczących się w przedziale ufności
        within_interval = ((group['True_Prob'] >= group['DOWN']) & (group['True_Prob'] <= group['UP'])).sum()
        total = group.shape[0]
        coverage = within_interval / total
        coverage_list.append(coverage)
    
    # Obliczanie minimalnego pokrycia
    min_coverage = min(coverage_list)
    
    return min_coverage



##############################  Wywoływanie funkcji ####################################




plot_correlation_matrix(feat_cal_imput_mean, title='Correlation Matrix for data_cal_imput_mean')

plot_correlation_matrix(feat_cal_noimput, title='Correlation Matrix for data_cal_noimput')

summary = missing_data_summary(feat_cal)
#print(summary)

data_val_imput_mean['True_Prob'] = uncertainty_knn(data_cal_imput_mean, data_val_imput_mean)




# Parametry
alpha = 0.1
n_neigh = 100

# Wizualizacja
data_cal_imput_mean_na = data_cal_imput_mean.copy()
data_val_imput_mean_na = data_val_imput_mean.copy()



data_cal_with_distances = calculate_nearest_neighbors_between_dfs(data_cal_imput_mean, data_cal_imput_mean, n_neigh)
data_val_with_distances = calculate_nearest_neighbors_between_dfs(data_cal_imput_mean, data_val_imput_mean, n_neigh)
data_val_with_distances = uncertainty_interval(data_cal_with_distances, data_val_with_distances, alpha)

data_cal_with_na = uncertainty_na(data_cal_imput_mean_na)
data_val_with_na = uncertainty_na(data_val_imput_mean_na)
data_val_with_na = uncertainty_interval(data_cal_with_na, data_val_with_na, alpha)


data_cal_with_mean = average_u_column(data_cal_with_distances, data_cal_with_na)
data_val_with_mean = average_u_column(data_val_with_distances, data_val_with_na)
data_val_with_mean = uncertainty_interval(data_cal_with_mean, data_val_with_mean, alpha)

draw_plot(data_val_with_distances)
print(feature_stratified_coverage_metric(data_val_with_distances))


draw_plot(data_val_with_na)
print(feature_stratified_coverage_metric(data_val_with_na))

draw_plot(data_val_with_mean)
print(feature_stratified_coverage_metric(data_val_with_mean))




