import pandas as pd
import numpy as np
import pickle
import time
import os
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# data function required for Hyperas hyperparameter tuning
def data(ticker, date):
    
    dataset = pd.read_csv(f'dataset/{ticker}_features.csv', index_col=0)
    dataset = dataset[dataset.index <= date]
    X = dataset[dataset.columns.to_list()[1:]]
    y = dataset[['label']]
    
    #X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.3, shuffle=False, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    #X_test = scaler.transform(X_test)
    
    return X_train, X_val, y_train, y_val #X_train, X_val, X_test, y_train, y_val, y_test
    
# model creation for hyperparameter search
def model(X_train, y_train, X_val, y_val):
    
    # network with hyperparameter search space
    # https://www.kaggle.com/kt66nf/hyperparameter-optimization-using-keras-hyperas
    
    # hidden layerns, units, dropout rate
    model = Sequential()
    model.add(Dense(input_shape=(100,), units={{choice(list(range(50, 201, 5)))}}, activation='relu', name='fc_layer_1'))
    model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_1'))
    
    choiceval = {{choice([2,3,4,5,6,7])}}
    activator = {{choice(['relu', 'tanh'])}}
    
    if choiceval == 2:
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_2'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_2'))
        
    if choiceval == 3:
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_2'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_2'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_3'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_3'))
        
    if choiceval == 4:
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_2'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_2'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_3'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_3'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_4'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_4'))
        
    if choiceval == 5:
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_2'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_2'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_3'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_3'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_4'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_4'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_5'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_5'))
        
    if choiceval == 6:
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_2'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_2'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_3'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_3'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_4'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_4'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_5'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_5'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_6'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_6'))
        
    if choiceval == 7:
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_2'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_2'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_3'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_3'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_4'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_4'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_5'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_5'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_6'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name='dropout_6'))
        model.add(Dense(units={{choice(list(range(50, 201, 5)))}}, activation=activator, name='fc_layer_7'))
        model.add(Dropout(rate={{uniform(0, 0.5)}}, name ='dropout_7'))
    
    # output layer
    model.add(Dense(units=1, activation='sigmoid', name='ouput'))

    # Early Stopping on Loss or Accuracy: https://datascience.stackexchange.com/questions/37186/early-stopping-on-validation-loss-or-on-accuracy
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.10, patience=15, min_lr=0.00005)
    callbacks_list= [es, reduce_lr]
    
    # learning rate
    opt_choice = {{choice(['Adam','SGD'])}}
    lr_pow = {{uniform(-5.75, -2.95)}}
    if opt_choice == 'Adam':
        model.compile(optimizer=Adam(learning_rate=math.pow(10, lr_pow)), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
    if opt_choice == 'SGD':
        model.compile(optimizer=SGD(learning_rate=math.pow(10, lr_pow)), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
    
    # epochs, batch size
    batch_size = {{choice(list(range(32, 257, 32)))}}
    
    # returning training history: https://github.com/maxpumperla/hyperas/issues/158
    history = model.fit(X_train, y_train,
                        epochs=250,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list, 
                        verbose=2)

    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f'Test accuracy: {acc:.4f}\n')
    return {'loss': loss, 'status': STATUS_OK, 'model': model}
    
    # return {'loss': loss, 'acc': -acc, 'status': STATUS_OK, 'model': model,  
    #         'history.val_loss':history.history['val_loss'], 'history.val_acc': history.history['val_accuracy'],
    #         'history.loss': history.history['loss'], 'history.acc': history.history['accuracy']}

def plot_tpe_losses(tpe_losses, save_name=None):
    import pandas as pd
    best_loss =  1
    best_losses = []
    for loss in tpe_losses:
        if loss < best_loss:
            best_losses.append(loss)
            best_loss = loss
        else:
            best_losses.append(best_loss)
            
    val_loss_df = pd.DataFrame({'Trial': list(range(1, len(tpe_losses)+1)),'loss':tpe_losses, 'minimum-loss': best_losses})
    ax = val_loss_df.plot.scatter(x='Trial', y='loss', figsize=(10,8), title='Model Losses over TPE trials');
    plt.plot(val_loss_df['minimum-loss'], '--', color='red')
    plt.legend()
    if save_name:
        ax.figure.savefig(save_name)


start = time.time()

# Trials() stores the history (return value from model()) of each TPE run
trials = Trials()
best_run, best_model, search_space = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=500,
                                      trials=trials, 
                                      eval_space=True,
                                      return_space=True,
                                      data_args=('AAPL','2011-01-28'),
                                      notebook_name='network')
end = time.time()
print(f"Time taken for evaluation: {end-start:.3f} seconds.")


from hyperas.utils import eval_hyperopt_space
print("----------TPE trials-------------")
nb_trials = len(trials)
losses = []
#models = []
for trial in trials.trials:

    #print(trial.get('result'))
    #best_val_loss = min(trial.get('result')['history.val_loss'])
    #val_losses.append(best_val_loss)
    
    losses.append(trial.get('result')['loss'])
    #models.append(trial.get('result')['model'])

with open(f'results/tpe_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)