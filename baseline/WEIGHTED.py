import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import random
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
import numpy as np

pd.set_option('display.max_columns', None)

PATH = '/Volumes/Transcend/Semester 3/Statistical Machine Learning/Assignments/Assignment 2/Data-Project2/'

PER_DEV = 0.3
PER_TEST = 0.333
SAMPLES = 100
SAMPLES_DEV = 100
FOLDS = 20
ITER = 10

enc = OneHotEncoder(handle_unknown='ignore')

def read_file(PATH):
    female = pd.read_csv(PATH + "FEMALE.csv")
    male = pd.read_csv(PATH + "MALE.csv")
    mixed = pd.read_csv(PATH + "MIXED.csv")

    female['type'] = 'female'
    male['type'] = 'male'
    mixed['type'] = 'mixed'

    data = pd.concat([female,male])
    data = pd.concat([data,mixed])
    return data


def encode_data(data):
    categorical_data = data[['Year', 'VR Band of Student', 'Ethnic group of student', 'School denomination']]
    categorical_data_encode = pd.get_dummies(categorical_data, columns=['Year', 'VR Band of Student', 'Ethnic group of student', 'School denomination'], drop_first=True)

    numerical_data = data[['FSM', 'VR1 Band']]
    numerical_data_scaled = preprocessing.scale(numerical_data)
    numerical_data_scaled = pd.DataFrame(numerical_data_scaled)
    numerical_data_scaled.columns = numerical_data.columns
    numerical_data_scaled.index = numerical_data.index

    type = data[['type']]
    response = data[['Exam Score']]
    final = pd.concat([numerical_data_scaled, categorical_data_encode, response, type], axis=1)

    return final


def partition_data(data, percentage_dev, percentage_test):
    X = data.drop(columns=['Exam Score'])
    y = data[['Exam Score']]
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=percentage_dev)
    X_test, X_dev, y_test, y_dev = train_test_split(X_other, y_other, test_size=percentage_test)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    dev = pd.concat([X_dev, y_dev], axis=1)

    return train, dev, test


def k_fold_plit(data, folds, size):
    fold = list()
    for i in range(0, folds):
        sample = data.sample(n = size)
        fold.append(sample)
    return fold


def cross_validation_set(replica_number, input, weight, percentage_dev = PER_DEV, percentage_test = PER_TEST, samples = SAMPLES, folds = FOLDS):
    female = input[input['type'] == 'female']
    male = input[input['type'] == 'male']
    mixed = input[input['type'] == 'mixed']

    if replica_number == 'female':
        train_female, dev_female, test_female = partition_data(female, percentage_dev, percentage_test)
        max = (male.shape[0] + mixed.shape[0])*3
        samples_target = int(max*weight)
        sample_domain = int(max*(1 - weight)/2)

        train = pd.concat([train_female.sample(n = samples).sample(n = samples_target, replace = True),male.sample(n = sample_domain, replace = True),mixed.sample(n = sample_domain, replace = True)])
        dev = k_fold_plit(dev_female, folds, SAMPLES_DEV)
        test = test_female
    elif replica_number == 'male':
        train_male, dev_male, test_male = partition_data(male, percentage_dev, percentage_test)
        max = (female.shape[0] + mixed.shape[0])*3
        samples_target = int(max*weight)
        sample_domain = int(max*(1 - weight)/2)

        train = pd.concat([train_male.sample(n = samples).sample(n = samples_target, replace = True),female.sample(n = sample_domain, replace = True),mixed.sample(n = sample_domain, replace = True)])
        dev = k_fold_plit(dev_male, folds, SAMPLES_DEV)
        test = test_male
    else:
        train_mixed, dev_mixed, test_mixed = partition_data(mixed, percentage_dev, percentage_test)
        max = (female.shape[0] + male.shape[0])*3
        samples_target = int(max*weight)
        sample_domain = int(max*(1 - weight)/2)

        train = pd.concat([train_mixed.sample(n = samples).sample(n = samples_target, replace = True),female.sample(n = sample_domain, replace = True),male.sample(n = sample_domain, replace = True)])
        dev = k_fold_plit(dev_mixed, folds, SAMPLES_DEV)
        test = test_mixed

    return train, dev, test


def linear_regression(target,input, iter = ITER):
    summary = pd.DataFrame()

    for i in range(1,iter + 1):
        alpha = random.uniform(0.0001, 0.9)
        weight = random.choice(np.arange(0.05, 1, 0.05))
        train, dev, test = cross_validation_set(target, input, weight)
        X_train = train.drop(columns=['type','Exam Score'])
        y_train = train[['Exam Score']]

        clf = linear_model.Lasso(random_state=0, alpha = alpha).fit(X_train, y_train)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('Logistic Regression', alpha, weight, mse, i)
            output = pd.DataFrame([out], columns=('model', 'alpha','weight', 'mse', 'iter'))
            summary = summary.append(output)
    summary = pd.DataFrame(summary.groupby(['model', 'alpha','weight', 'iter']).mean().sort_values(by=['mse']).reset_index())

    best_alpha = summary['alpha'].iloc[0]
    best_weight = summary['weight'].iloc[0]
    train, dev, test = cross_validation_set(target, input, best_weight)
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    clf = linear_model.Lasso(random_state=0, alpha = best_alpha).fit(X_train, y_train)
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('Logistic Regression',{'alpha':round(best_alpha,3), 'weight':round(best_weight,3)}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse


def neural_network(target,input, iter = ITER):
    summary = pd.DataFrame()

    for i in range(1,iter + 1):
        hidden_layer_sizes = random.choice([(random.randint(1, 20), random.randint(1, 20), random.randint(1, 20),), (random.randint(1, 20), random.randint(1, 20),), (random.randint(1, 20),)])
        alpha = random.uniform(0.0001, 0.9)
        weight = random.choice(np.arange(0.05, 1, 0.05))
        train, dev, test = cross_validation_set(target, input, weight)
        X_train = train.drop(columns=['type','Exam Score'])
        y_train = train[['Exam Score']]

        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', alpha = alpha, max_iter=1000)
        mlp.fit(X_train, y_train.values.ravel())
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = mlp.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('Neural Network', hidden_layer_sizes, alpha , weight, mse, i)
            output = pd.DataFrame([out], columns=('model', 'hidden_layers', 'alpha', 'weight', 'mse', 'iter'))
            summary = summary.append(output)
    summary = pd.DataFrame(summary.groupby(['model', 'hidden_layers', 'alpha', 'weight', 'iter']).mean().sort_values(by=['mse']).reset_index())

    best_hidden = summary['hidden_layers'].iloc[0]
    best_alpha = summary['alpha'].iloc[0]
    best_weight = summary['weight'].iloc[0]
    train, dev, test = cross_validation_set(target, input, best_weight)
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    mlp = MLPRegressor(hidden_layer_sizes=best_hidden, activation='relu', alpha = best_alpha, max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    y_pred = mlp.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('Neural Network', {'hidden_layer_sizes':best_hidden, 'alpha': round(best_alpha,3) }, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse


def support_vector(target,input, iter = ITER):
    summary = pd.DataFrame()

    for i in range(1,iter + 1):
        epsilon = random.uniform(0.0001, 0.9)
        C = random.uniform(0.001, 10)
        kernel = random.choice(['linear', 'rbf'])
        weight = random.choice(np.arange(0.05, 1, 0.05))
        train, dev, test = cross_validation_set(target, input, weight)
        X_train = train.drop(columns=['type','Exam Score'])
        y_train = train[['Exam Score']]

        clf = SVR(C=C, epsilon=epsilon, kernel=kernel).fit(X_train, y_train)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('SVM', epsilon, C, kernel, weight, mse, i)
            output = pd.DataFrame([out], columns=('model', 'epsilon', 'C', 'kernel','weight', 'mse', 'iter'))
            summary = summary.append(output)
    summary = pd.DataFrame(summary.groupby(['model', 'epsilon', 'C', 'kernel','weight', 'iter']).mean().sort_values(by=['mse']).reset_index())

    best_epsilon = summary['epsilon'].iloc[0]
    best_C = summary['C'].iloc[0]
    best_kernel = summary['kernel'].iloc[0]
    best_weight = summary['weight'].iloc[0]
    train, dev, test = cross_validation_set(target, input, best_weight)
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    clf = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel).fit(X_train, y_train)
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('SVM',{'epsilon':round(best_epsilon,3), 'C':round(best_C,3), 'kernel': best_kernel, 'weight':round(best_weight,3)}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse


def cross_validation(input):
    summary_final = pd.DataFrame()
    for i in ['female', 'male','mixed']:

        mse = linear_regression(i, input)
        mse['target'] = i
        summary_final = summary_final.append(mse)

        mse = neural_network(i, input)
        mse['target'] = i
        summary_final = summary_final.append(mse)

        mse = support_vector(i, input)
        mse['target'] = i
        summary_final = summary_final.append(mse)
    return summary_final


def main():
    input = read_file(PATH)
    input = encode_data(input)
    summary = cross_validation(input)
    summary['algorithm'] = 'WEIGHTED_' + str(SAMPLES)
    summary.to_csv('summary_WEIGHTED_' + str(SAMPLES) + '.csv', index=False, sep = ',')


if __name__ == '__main__':
    main()
