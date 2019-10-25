import numpy as np
from keras import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.layers import Embedding, Flatten, Input, Dense, Subtract, LSTM, concatenate, GlobalAveragePooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA


def read_data(filename):
    data = read_csv(filename, header=None, names=list('abcdefghij'),
                    dtype=dict(zip(list('abcdefghij'), [int] + [str] * 4 + [int] * 3 + [str] * 2)))
    return data

def clean_data(data):
    conditions = [
        (data['i'] == 'W'),
        (data['i'] == 'L'),
        (data['i'] == 'D')]
    choices = [1, 0, 0.5]
    data['wdb'] = np.select(conditions, choices)
    data = data[data.i != 'D']

    won = len(data[data['i'] == "W"])
    lost = len(data[data['i'] == "L"])
    total = won+lost
    print("Won:", won, won/total*100, ", Lost:", lost, lost/total*100)

    data = data.to_numpy()
    return data

def model():
    # data = read_data('../bet_ISDB_Leagues.csv')
    data = read_data('../test_baseline.csv')

    data = clean_data(data)
    size = data.size
    # print(size)

    teams = np.unique(data[:, 3:4])
    n_teams = teams.size
    X = data[:, [3,4]]
    # print(np.shape(X))
    y = data[:, 10]

    X = X.flatten()
    label_encoder = LabelEncoder()
    X = label_encoder.fit_transform(X)
    teams_encoded = label_encoder.fit_transform(teams)
    print(X)
    X = np.reshape(X, (-1,2))

    separator = int(X.__len__()*0.8)
    X_train = X[:separator]
    X_test = X[separator:]
    y_train = y[:separator]
    y_test = y[separator:]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(np.shape(X_test))
    # integer encode
    """Embedding model"""
    input = Input(shape=(1,))
    embed_layer = Embedding(input_dim=n_teams, input_length=1, output_dim=3, name='Team-Strength')
    em_tensor = embed_layer(input)
    flat_tensor = Flatten()(em_tensor)
    model = Model(input, flat_tensor)

    # model = Sequential()
    # model.add(Embedding(input_dim=n_teams, input_length=1, output_dim=16, name='Team-Strength'))
    # # model.add(GlobalAveragePooling1D())
    # model.add(Flatten())

    """Predicting Model"""
    # input = Input(shape=(1,))
    input_tensor_1 = Input((1,))
    input_tensor_2 = Input((1,))
    output_tensor_1 = model(input_tensor_1)
    output_tensor_2 = model(input_tensor_2)
    # x = Subtract()([output_tensor_1, output_tensor_2])

    x = concatenate([output_tensor_1, output_tensor_2], axis=-1)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dense(16, activation='relu')(x)
    # x = Dense(8, activation='relu')(x)
    # x = Dense(4, activation='relu')(x)
    x = Dense(64, activation='selu')(x)
    x = Dense(32, activation='selu')(x)
    x = Dense(16, activation='selu')(x)
    x = Dense(8, activation='selu')(x)
    x = Dense(4, activation='selu')(x)

    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_tensor_1, input_tensor_2], outputs=predictions)
    # opt = SGD(lr=0.01, momentum=0.9)
    # opt = 'rmsprop'
    opt = 'adam'
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, validation_split=0.10, epochs=10, batch_size=32)

    embedded = Model(input=input, output=em_tensor)
    output = embedded.predict(teams_encoded)
    print(output)
    print(np.shape(output))
    output = np.reshape(output, (36,3))
    pca_embedding(output, teams)
    # o = list(zip(teams, output[:, 0]))
    # p = pd.DataFrame(o, columns=list('ab'))
    # p.sort_values(by='b', inplace=True, ascending=False)
    # print(p)
    # p.to_csv('output_ranking_table.csv')
    # ranking_table(data)

    loss, accuracy = model.evaluate([X_test[:,0], X_test[:,1]], y_test, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()

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
    df = pd.DataFrame({"Name":teams.tolist()})
    df['W'] = 0
    df['L'] = 0
    for i in range(np.shape(data)[0]):
        if(data[i,8] == 'W'):

            df.loc[df['Name'] == data[i,3], "W"] += 1
            df.loc[df['Name'] == data[i,4], "L"] += 1

            # df['W'][df.iloc[df['Name'] == data[i,3]]] += 1
            # df['L'][df.iloc[df['Name'] == data[i,4]]] += 1
        else:
            df.loc[df['Name'] == data[i,3], "L"] += 1
            df.loc[df['Name'] == data[i,4], "W"] += 1
    df['Dif'] = df['W'] - df['L']
    df.sort_values(by = 'Dif', inplace=True, ascending=False)
    # print(df)
    df.to_csv('ranking_table.csv')

if __name__ == '__main__':
    # print(np.unique(data[:,3]))

    model()
