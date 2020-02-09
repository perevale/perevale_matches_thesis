import csv
import operator

import numpy as np
from keras import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
from keras.layers import Embedding, Flatten, Input, Dense, Subtract, LSTM, concatenate, GlobalAveragePooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.decomposition import PCA
import math
from sacred import Experiment
ex = Experiment('model')
from sacred.observers import MongoObserver

import warnings
warnings.filterwarnings("ignore")

ex.observers.append(MongoObserver(url='localhost:27017',
                                  db_name='sacred'))
def read_data(filename):
    names = ['year', 'league', 'time','home_team','away_team', 'home_score','away_score','difference_score','result', 'country']
    """Read the data from csv with correct data types."""
    data = read_csv(filename, header=None, names=names,
                    dtype=dict(zip(names, [int] + [str] * 4 + [int] * 3 + [str] * 2)))
    return data


def clean_data(data: pd.DataFrame, convert_to_numpy=True, allow_draw=False) -> np.array:
    """Add a column to transform result of the match into int, """
    # result ot int
    conditions = [
        (data['result'] == 'W'),
        (data['result'] == 'L'),
        (data['result'] == 'D')]
    choices = [1, 0, 2]
    data['wdb'] = np.select(conditions, choices)

    # ignore the draw results
    if not allow_draw:
        data = data[data['result'] != 'D']

    # print some metadata
    won = len(data[data['result'] == "W"])
    lost = len(data[data['result'] == "L"])
    draw = len(data[data['result'] == "D"])
    total = len(data)
    # print("Won:", won, won / total * 100, ", Lost:", lost, lost / total * 100)
    print("Won: {}%, Lost: {}%, Draw: {}%".format(won/total, lost/total, draw/total))

    # convert to numpy
    if convert_to_numpy:
        data = data.to_numpy()

    return data

def create_model(n_teams, result , activation='selu', loss='binary_crossentropy', optimizer='adam'):
    """Embedding model"""
    input = Input(shape=(1,))
    embed_layer = Embedding(input_dim=n_teams, input_length=1, output_dim=3, name='Team-Strength')
    em_tensor = embed_layer(input)
    flat_tensor = Flatten()(em_tensor)
    em_model = Model(input, flat_tensor)

    # em_model = Sequential()
    # em_model.add(Embedding(input_dim=n_teams, input_length=1, output_dim=16, name='Team-Strength'))
    # # # model.add(GlobalAveragePooling1D())
    # em_model.add(Flatten())

    """Predicting Model"""
    # input = Input(shape=(1,))
    input_tensor_1 = Input((1,))
    input_tensor_2 = Input((1,))
    output_tensor_1 = em_model(input_tensor_1)
    output_tensor_2 = em_model(input_tensor_2)

    # x = Subtract()([output_tensor_1, output_tensor_2])

    x = concatenate([output_tensor_1, output_tensor_2], axis=-1)
    x = Dense(128, activation=activation)(x)
    # x = Dense(64, activation=activation)(x)
    # x = Dense(32, activation=activation)(x)
    # x = Dense(16, activation=activation)(x)
    x = Dense(8, activation=activation)(x)
    # x = Dense(4, activation=activation)(x)

    predictions = Dense(result, activation='sigmoid')(x)
    model = Model(inputs=[input_tensor_1, input_tensor_2], outputs=predictions)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['categorical_accuracy'])
                  # metrics=['accuracy'])
    return model#, em_tensor

@ex.config
def cfg():
  C = 1.0
  gamma = 0.7

def prepare_data(data):
    teams = np.unique(data[:, 3:4])
    X = data[:, [3, 4]]
    # print(np.shape(X))
    y = data[:, 10]

    X = X.flatten()
    label_encoder = LabelEncoder()
    X = label_encoder.fit_transform(X)
    teams_encoded = label_encoder.fit_transform(teams)
    # print(X)
    X = np.reshape(X, (-1, 2))

    separator = int(X.__len__() * 0.8)
    X_train = X[:separator]
    X_test = X[separator:]
    y_train = y[:separator]
    y_test = y[separator:]
    return X_train, X_test, y_train, y_test

def grid_search(data, output):
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    activation = ['elu', 'relu', 'selu', 'softplus','tanh', 'sigmoid','hard_sigmoid','exponential','linear']
    loss = ['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error','squared_hinge','hinge','categorical_hinge','logcosh','categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']
    d = {'Optimizer': ['a'], 'Activation': ['a'], 'Loss':['a'], 'Accuracy':[0]}
    header = True
    # df = pd.DataFrame(data=d)
    best_acc=0
    best_values=[]
    for opt in optimizer:
        for act in activation:
            for l in loss:
                print("Training:", opt,act,l)
                test_accuracy = run_model(data, activation= act, loss=l, opt=opt)
                # vals = opt+'_'+act+'_'+l
                d1 = {'Optimizer': [opt], 'Activation': [act], 'Loss':[l], 'Accuracy':[test_accuracy]}
                df = pd.DataFrame(data=d1)
                # df = df.append(df2, ignore_index=True)

                mode = 'w' if header else 'a'
                df.to_csv(output, index=None, mode=mode, header=header)
                header = False
                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    best_values = [opt,act,l]
                    print(best_acc, best_values)
    # df = df.iloc[1:]
    return best_acc,best_values
@ex.capture
def run_model(data: np.array, result, activation='selu', loss='categorical_crossentropy', opt = 'adam'):
    # data = read_data(input_file)  # pandas dataframe
    # data = clean_data(data)  # numpy array

    teams = np.unique(data[:, 3:4])
    n_teams = teams.size

    onehot_encoder = OneHotEncoder(sparse=False)
    X_train, X_test, y_train, y_test = prepare_data(data)



    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # print(np.shape(X_test))
    # integer encode

    model= create_model(n_teams, result, activation, loss, opt)

    # opt = SGD(lr=0.01, momentum=0.9)
    # opt = 'rmsprop'



    history = model.fit([X_train[:, 0], X_train[:, 1]], onehot_encoder.fit_transform(y_train.reshape(-1,1)), validation_split=0.10, epochs=5, batch_size=32, verbose=0)


    # print(model.layers[2])
    # embedded = Model(input=Input(shape=(1,)), output=em_tensor)
    # output = embedded.predict(teams_encoded)
    # print(output)
    # print(np.shape(output))
    # output = np.reshape(output, (n_teams, -1))

    # o = list(zip(teams, output[:, 0]))
    # p = pd.DataFrame(o, columns=list('ab'))
    # p.sort_values(by='b', inplace=True, ascending=False)
    # print(p)
    # p.to_csv('output_ranking_table.csv')
    # ranking_table(data)
    predictions = model.predict([X_test[:, 0], X_test[:, 1]])
    ground_truth = onehot_encoder.fit_transform(y_test.reshape(-1,1))
    acc = np.sum(np.logical_not(np.argmax(predictions, axis=1) == np.argmax(ground_truth, axis=1)))/np.shape(y_test)[0]

    loss, accuracy = model.evaluate([X_test[:, 0], X_test[:, 1]], onehot_encoder.fit_transform(y_test.reshape(-1,1)), verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    print("My accuracy =", acc, "Keras accuracy =", accuracy)

    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xticks(range(len(history.history['loss'])),
               range(1, len(history.history['loss']) + 1))

    plt.xlabel('epochs')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['categorical_accuracy'], label='train')
    plt.plot(history.history['val_categorical_accuracy'], label='test')
    plt.xlabel('epochs')
    plt.xticks(range(len(history.history['categorical_accuracy'])), range(1,len(history.history['categorical_accuracy'])+1))
    plt.legend()
    plt.show()

    return accuracy



    # pca_embedding(output, teams)



def pca_embedding(input, teams):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(input)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    # words = list(model.wv.vocab)
    for i, team in enumerate(teams):
        plt.annotate(team, xy=(principalComponents[i, 0], principalComponents[i, 1]), size=8)
    plt.show()


def ranking_table(data):
    teams = np.unique(data[:, 3:4])
    df = pd.DataFrame({"Name": teams.tolist()})
    df['W'] = 0
    df['L'] = 0

    for i in range(np.shape(data)[0]):
        if (data[i, 8] == 'W'):

            df.loc[df['Name'] == data[i, 3], "W"] += 1
            df.loc[df['Name'] == data[i, 4], "L"] += 1

            # df['W'][df.iloc[df['Name'] == data[i,3]]] += 1
            # df['L'][df.iloc[df['Name'] == data[i,4]]] += 1
        else:
            df.loc[df['Name'] == data[i, 3], "L"] += 1
            df.loc[df['Name'] == data[i, 4], "W"] += 1
    df['Dif'] = df['W'] - df['L']
    df.sort_values(by='Dif', inplace=True, ascending=False)
    # print(df)
    df.to_csv('ranking_table.csv')


def elo(rating_home, rating_away, delta, result, k = 30, c = math.e, gamma = 2, d = 400):
    """
    :param rating_home: rating of the home team
    :param rating_away: rating of the away team
    :param k: a learning rate
    :param c: metaparameter of the model
    :param gamma: metaparameter scaling the influence of the goal difference on the rating change
    :param delta: absolute goal difference
    :param d: metaparameter of the model
    :param result: home won = 1, home draw = 0.5, home loss = 0
    :return: updated rating_home, rating_away
    """

    E_h = 1.0 / (1 + math.pow(c, 1.0 * (rating_home - rating_away) / d))
    E_a = 1 - E_h

    rating_home = rating_home + k * math.pow((1 + delta), gamma) * (result - E_h)
    rating_away = rating_away - k * math.pow((1 + delta), gamma) * (result - E_h)

    return rating_home, rating_away

def ranking_elo(input_data, output_elo):
    ratings = {}

    with open(input_data, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            team_home = row[3]
            team_away = row[4]
            delta = abs(int(row[7]))
            result = row[8]
            if result == 'W':
                result = 1
            elif result == 'L':
                result = 0
            else:
                result = 0.5
            if team_home not in ratings:
                ratings[team_home] = 1000
            if team_away not in ratings:
                ratings[team_away] = 1000
            ratings[team_home], ratings[team_away] = elo(ratings[team_home], ratings[team_away], delta=delta,
                                                         result=result)
    ratings_sorted = sorted(ratings.items(), key=operator.itemgetter(1), reverse=True)
    with open(output_elo, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['name', 'rating'])
        for row in ratings_sorted:
            csv_out.writerow(row)

def sort_csv(input_file, output_file, column):
    with open(input_file, 'r') as csv_file, open(output_file, 'w+', newline='') as out:
        data = csv.reader(csv_file, delimiter=',')
        # data[1:] = sorted(data[1:], key=lambda x: x[column], reverse=True)
        headers = next(data, None)
        sorted_table = sorted(data, key=operator.itemgetter(column), reverse=True)
        csv_out = csv.writer(out)
        csv_out.writerow(headers)
        # csv_out.writerow(['name', 'rating'])
        for row in sorted_table:
            csv_out.writerow(row)


def train_all_leagues(input_data, input_leagues, result_leagues):
    all_data = read_data(input_data)
    accuracies = []
    with open(input_leagues, 'r') as csv_file, open(result_leagues, 'w+', newline='') as out:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print("--------------------------------------------------------------------------------------------")
            league = row[1]
            league_data = all_data[all_data['league'] == league]
            league_data = clean_data(league_data, allow_draw=True, convert_to_numpy=True)
            accuracy_league = run_model(league_data, 3)
            csv_out = csv.writer(out)
            csv_out.writerow([league, accuracy_league])
            accuracies.append(accuracy_league)
    mean = np.mean(accuracies)
    var = np.var(accuracies)
    return mean, var


@ex.automain
def main():
    input_data = r'bet_ISDB_Leagues.csv'
    input_leagues = r'league_code.csv'
    result_leagues = r'result_leagues.csv'

    mean, var = train_all_leagues(input_data, input_leagues, result_leagues)
    print("Mean accuracy for all leagues:", mean, "Variance accuracy for all leagues:", var)





    # input_data = r'./test_baseline.csv'
    # input = read_data(input_data)
    #
    # # data = clean_data(input, allow_draw=False, convert_to_numpy=True)
    # # run_model(data, 2)
    # data = clean_data(input, allow_draw=True, convert_to_numpy=True)
    # run_model(data, 3)

    # grid_search_file = 'grid_search.csv'
    # grid_sorted = 'grid_sorted.csv'
    # best_acc,best_values = grid_search(data,grid_search_file)
    # print(best_acc, best_values)
    # sort_csv(grid_search_file, grid_sorted, 3)



    # output_elo = r'./output_elo.csv'
    # ranking_elo(input_data, output_elo)
