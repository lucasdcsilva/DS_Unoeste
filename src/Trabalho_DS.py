import os
from numpy.core.fromnumeric import trace
from numpy.lib.utils import info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC




plt.style.use('ggplot')

# Leitura dos diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,'data')


# List Compreension do dataset
file_names = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]

# listando o dataset
for i in file_names:
    df = pd.read_csv(os.path.join(DATA_DIR, i))

# Iniciando o tratamento dos dados. True = 1 and False = 0
map_data = {True: 1, False: 0}
df['mask'] = df['mask'].map(map_data)
#print('\nAlteração de valores categóricos: \n,df.head(5))

# ******************* Amostras por classe **************************************************
sample0 = np.where(df.loc[df['mask'] == 0])
sample1 = np.where(df.loc[df['mask'] == 1])


# ******************* Quantidade de amostrar por classe **************************************
com_mascara = len(df.loc[df['mask'] == 1])
sem_mascara = len(df.loc[df['mask'] == 0])

# ******************* Dados Faltantes e valores = 0 **************************************************
dt_feature = df.iloc[:,:-1]
dt_target = df.iloc[:, -1]
dt_feature = dt_feature.mask(dt_feature == 0).fillna(dt_feature.mean())


# ******************* Plotando dados **************************************************

# Histograma de classes
def plot_hist():
    plt.hist(df.iloc[:,-1], color='b', width=.1)
    plt.xlabel('Quantidade de amostras')
    plt.ylabel('Histograma de Classes')
    plt.show()

# Histograma web offline
def target_count():
    trace = go.Bar(x = df['mask'].value_counts().values.tolist(),
                    y = ['Com mascara', 'Sem mascara'],
                    orientation = 'v',
                    text=df['mask'].value_counts().values.tolist(),
                    textfont=dict(size=15),
                    textposition = 'auto',
                    opacity = 0.8, marker=dict(color=['lightskyblue', 'gold'],
                    line=dict(color='#000000', width=1.5)))
    layout = dict(title='Resultado')
    fig = dict(data=[trace], layout=layout)
    py.plot(fig)

# Análise de correlação
def correlation(size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

# Boxplot
def boxplot():
    f, ax = plt.subplots(figsize=(11, 15))
    ax.set_facecolor('#fafafa')
    ax.set(xlim=(-0.5, 200))
    plt.ylabel('Variaveis')
    plt.title("Overview Dataset")
    ax = sns.boxplot(data=df, orient='v', palette='Set2')
    plt.show()

def histograma():
    df.hist()
    plt.show()

# ******************* Preparando os modelos de ML **************************************************

# Criar uma lista de armazenamento de accuracia
accuracy_PC = []
accuracy_NB = []

def split_model():
    print('\n ************************* Resultados ************************ \n')
    rould = 0.10
    epochs = 1
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3, random_state=i)
        print('Divisão do conjunto de dados\n')
        print('X_train: %d\ny_train: %d\nX_test: %d\ny_test: %d\n' %(len(X_train), len(y_train), len(X_test), len(y_test)))
        print('Quantidade de amostras da classe 0: ', len(y_train.loc[y_train == 0]))
        print('Quantidade de amostras da classe 1: ', len(y_train.loc[y_train == 1]))

        # Perceptron
        percep = Perceptron()
        percep.fit(X_train, y_train)
        percep.predictions = percep.predict(X_test)
        acc_perpep = percep.score(X_test, y_test)

        # Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb.predictions = gnb.predict(X_test)
        acc_nb = gnb.score(X_test, y_test)

        accuracy_PC.append(acc_perpep)
        accuracy_NB.append(acc_nb)

        print('\n Resultados Perceptron: \nAcc_Pc: ', acc_perpep)
        print('\n Resultados Naive Bayes: \nAcc_NB: ', acc_nb)
        print('Vetor de Acc Perceptron: ', accuracy_PC)
        print('Vetor de Acc Bayes: ', accuracy_NB)

def exemplos():
    X_train, X_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3, random_state=None, shuffle=True)
    seed=7
    scoring='accuracy'
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #Exibit plot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


# Chamando as funções
plot_hist()
histograma()
target_count()
correlation()
split_model()
exemplos()