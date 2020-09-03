import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import numpy as np
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 # import the two-stage algorithm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
import time
from sklearn.tree import DecisionTreeRegressor

PATH = '/Volumes/Transcend/Semester 3/Statistical Machine Learning/Assignments/Assignment 2/Data-Project2/'
ALGORITHMS = ['ALL']
MODELS = ['female', 'male', 'mixed']

PER_DEV = 0.3
PER_TEST = 0.333
FOLDS = 20
STEPS = 10
ITER = 10
LEARNING_RATE = 0.01
SAMPLES = np.arange(0.01, 0.52, 0.02)
#SAMPLES = np.arange(100, 1050, 100)
SAMPLES_DEV = 100

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

def cross_validation_set(replica_number, input, samples, percentage_dev = PER_DEV, percentage_test = PER_TEST, folds = FOLDS):
    female = input[input['type'] == 'female']
    male = input[input['type'] == 'male']
    mixed = input[input['type'] == 'mixed']

    if replica_number == 'female':
        train_female, dev_female, test_female = partition_data(female, percentage_dev, percentage_test)

        train_domain = pd.concat([male,mixed])
        sample_number = int(samples*train_domain.shape[0]/(1-samples))
        if sample_number <= train_female.shape[0]:
            train_target = train_female.sample(n = sample_number)
        else:
            remaining = sample_number - train_female.shape[0]
            remaining_target = train_female.sample(n = remaining, replace = True)
            train_target = pd.concat([train_female,remaining_target])
        train = pd.concat([train_domain,train_target])

        dev = k_fold_plit(dev_female, folds, SAMPLES_DEV)
        test = test_female
        sample_size = [male.shape[0] + mixed.shape[0], sample_number]

        weights_female = np.empty(sample_number, dtype=np.float64)
        weights_female[:] = 1. /sample_number

        weights_male = np.empty(male.shape[0], dtype=np.float64)
        weights_male[:] = 1. /male.shape[0]

        weights_mixed = np.empty(mixed.shape[0], dtype=np.float64)
        weights_mixed[:] = 1. /mixed.shape[0]

        weights = np.concatenate((weights_male, weights_mixed, weights_female))
        adj_weight = weights/weights.sum()
    elif replica_number == 'male':
        train_male, dev_male, test_male = partition_data(male, percentage_dev, percentage_test)

        train_domain = pd.concat([female,mixed])
        sample_number = int(samples*train_domain.shape[0]/(1-samples))
        if sample_number <= train_male.shape[0]:
            train_target = train_male.sample(n = sample_number)
        else:
            remaining = sample_number - train_male.shape[0]
            remaining_target = train_male.sample(n = remaining, replace = True)
            train_target = pd.concat([train_male,remaining_target])
        train = pd.concat([train_domain,train_target])

        dev = k_fold_plit(dev_male, folds, SAMPLES_DEV)
        test = test_male
        sample_size = [female.shape[0] + mixed.shape[0], sample_number]

        weights_male = np.empty(sample_number, dtype=np.float64)
        weights_male[:] = 1. /sample_number

        weights_female = np.empty(female.shape[0], dtype=np.float64)
        weights_female[:] = 1. /female.shape[0]

        weights_mixed = np.empty(mixed.shape[0], dtype=np.float64)
        weights_mixed[:] = 1. /mixed.shape[0]

        weights = np.concatenate((weights_female, weights_mixed, weights_male))
        adj_weight = weights/weights.sum()
    else:
        train_mixed, dev_mixed, test_mixed = partition_data(mixed, percentage_dev, percentage_test)

        train_domain = pd.concat([female,male])
        sample_number = int(samples*train_domain.shape[0]/(1-samples))
        if sample_number <= train_mixed.shape[0]:
            train_target = train_mixed.sample(n = sample_number)
        else:
            remaining = sample_number - train_mixed.shape[0]
            remaining_target = train_mixed.sample(n = remaining, replace = True)
            train_target = pd.concat([train_mixed,remaining_target])
        train = pd.concat([train_domain,train_target])

        dev = k_fold_plit(dev_mixed, folds, SAMPLES_DEV)
        test = test_mixed
        sample_size = [female.shape[0] + male.shape[0], samples]

        weights_mixed = np.empty(sample_number, dtype=np.float64)
        weights_mixed[:] = 1. /sample_number

        weights_female = np.empty(female.shape[0], dtype=np.float64)
        weights_female[:] = 1. /female.shape[0]

        weights_male = np.empty(male.shape[0], dtype=np.float64)
        weights_male[:] = 1. /male.shape[0]

        weights = np.concatenate((weights_female, weights_male, weights_mixed))
        adj_weight = weights/weights.sum()

    return train, dev, test, sample_size, adj_weight

def all(train, dev, test, sample_size, adj_weight):
    summary_final = pd.DataFrame()
    all_data = pd.DataFrame()

    mse, summary = linear_regression(train, dev, test, sample_size, adj_weight)
    all_data = all_data.append(summary)
    summary_final = summary_final.append(mse)

    #mse, summary = neural_network(train, dev, test, sample_size, adj_weight)
    #all_data = all_data.append(summary)
    #summary_final = summary_final.append(mse)

    #mse, summary = support_vector(train, dev, test, sample_size, adj_weight)
    #all_data = all_data.append(summary)
    #summary_final = summary_final.append(mse)

    mse, summary = decision_tree(train, dev, test, sample_size, adj_weight)
    all_data = all_data.append(summary)
    summary_final = summary_final.append(mse)

    return all_data, summary_final

################################################################################
## Linear Regression
################################################################################

def linear_regression(train, dev, test, sample_size, adj_weight,iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        start = time.time()
        alpha = random.uniform(0.0001, 0.9)
        clf = TwoStageTrAdaBoostR2(linear_model.Lasso(random_state=0, alpha = alpha), sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
        clf.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('Linear Regression', alpha, mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'alpha', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
            end = time.time()
            print('LR', i, j, end - start)
    summary_final = pd.DataFrame(summary.groupby(['model', 'alpha', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_alpha = summary_final['alpha'].iloc[0]
    clf = TwoStageTrAdaBoostR2(linear_model.Lasso(random_state=0, alpha = best_alpha), sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
    clf.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('Linear Regression',{'alpha':round(best_alpha,3)}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    end = time.time()
    print('fin_LR', end - start)
    return mse, summary

################################################################################
## Neural Network
################################################################################

def neural_network(train, dev, test, sample_size, adj_weight,iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        start = time.time()
        hidden_layer_sizes = random.choice([(random.randint(1, 100), random.randint(1, 100), random.randint(1, 100),), (random.randint(1, 100), random.randint(1, 100),), (random.randint(1, 100),)])
        alpha = random.uniform(0.0001, 0.9)
        mlp = TwoStageTrAdaBoostR2(MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', alpha = alpha, max_iter=2000), sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
        mlp.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = mlp.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('Neural Network', hidden_layer_sizes, alpha , mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'hidden_layers', 'alpha', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
            end = time.time()
            print('LR', i, j, end - start)
    summary_final = pd.DataFrame(summary.groupby(['model', 'hidden_layers', 'alpha', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_hidden = summary_final['hidden_layers'].iloc[0]
    best_alpha = summary_final['alpha'].iloc[0]
    mlp = MLPRegressor(hidden_layer_sizes=best_hidden, activation='relu', alpha = best_alpha, max_iter=2000)
    mlp.fit(X_train, y_train.values.ravel())
    y_pred = mlp.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('Neural Network', {'hidden_layer_sizes':best_hidden, 'alpha': round(best_alpha,3) }, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    return mse


################################################################################
## Support Vector Machine
################################################################################

def support_vector(train, dev, test, sample_size, adj_weight,iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        start = time.time()
        epsilon = random.uniform(0.0001, 0.9)
        C = random.uniform(0.001, 10)
        kernel = random.choice(['linear', 'rbf'])
        clf = TwoStageTrAdaBoostR2(SVR(C=C, epsilon=epsilon, kernel=kernel), sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
        clf.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('SVM', epsilon, C, kernel, mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'epsilon', 'C', 'kernel', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
            end = time.time()
            print('SVM', i, j, end - start)
    summary_final = pd.DataFrame(summary.groupby(['model', 'epsilon', 'C', 'kernel', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_epsilon = summary_final['epsilon'].iloc[0]
    best_C = summary_final['C'].iloc[0]
    best_kernel = summary_final['kernel'].iloc[0]
    clf = TwoStageTrAdaBoostR2(SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel), sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
    clf.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('SVM',{'epsilon':round(best_epsilon,3), 'C':round(best_C,3), 'kernel': best_kernel}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    end = time.time()
    print('fin_SVM', end - start)
    return mse, summary


################################################################################
## Decision Tree Regressor
################################################################################

def decision_tree(train, dev, test, sample_size, adj_weight,iter = ITER):
    summary = pd.DataFrame()
    X_train = train.drop(columns=['type','Exam Score'])
    y_train = train[['Exam Score']]
    X_test = test.drop(columns=['type','Exam Score'])
    y_test = test[['Exam Score']]
    for i in range(1,iter + 1):
        start = time.time()
        max_depth = random.choice([5, 10, 15, 20, 25, 30, None])
        min_samples_leaf = random.choice([1, 2, 4])
        min_samples_split = random.choice([2, 5, 10])
        clf = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,  min_samples_split=min_samples_split),
                                   sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
        clf.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
        for j in range(0, len(dev)):
            dev_df = dev[j]
            X_dev = dev_df.drop(columns=['type','Exam Score'])
            y_dev = dev_df[['Exam Score']]
            y_pred_cv = clf.predict(X_dev).round(decimals=0)
            mse = mean_squared_error(y_dev, y_pred_cv)
            out = ('DT', max_depth, min_samples_leaf, min_samples_split, mse, i, j)
            output = pd.DataFrame([out], columns=('model', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'mse', 'iter', 'fold'))
            summary = summary.append(output)
            end = time.time()
            print('DT', i, j, end - start)
    summary_final = pd.DataFrame(summary.groupby(['model', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'iter']).mean().sort_values(by=['mse']).reset_index())
    best_max_depth = summary_final['max_depth'].iloc[0]
    best_min_samples_leaf = summary_final['min_samples_leaf'].iloc[0]
    best_min_samples_split = summary_final['min_samples_split'].iloc[0]
    clf = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf,  min_samples_split=best_min_samples_split),
                                   sample_size = sample_size, steps = STEPS, learning_rate = LEARNING_RATE, fold = FOLDS)
    clf.fit(X_train.to_numpy(), np.array(y_train['Exam Score'].values.tolist()), sample_weight = adj_weight)
    y_pred = clf.predict(X_test).round(decimals=0)
    mse = mean_squared_error(y_test, y_pred)
    out = ('DT',{'max_depth':best_max_depth, 'min_samples_leaf':min_samples_leaf, 'min_samples_split': min_samples_split}, mse)
    mse = pd.DataFrame([out], columns=('model','parameters','mse'))
    end = time.time()
    print('fin_DT', end - start)
    return mse, summary


def main():
    input = read_file(PATH)
    input = encode_data(input)

    summary_final = pd.DataFrame()
    all_data_final = pd.DataFrame()
    for model in MODELS:
        summary = pd.DataFrame()
        all_data_summary = pd.DataFrame()
        for samples in SAMPLES:
            train, dev, test, sample_size, adj_weight = cross_validation_set(model, input, samples)
            all_data, mse = all(train, dev, test, sample_size, adj_weight)
            mse['samples'] = samples
            all_data['samples'] = samples

            summary = summary.append(mse)
            all_data_summary = all_data_summary.append(all_data)
        summary['target'] = model
        all_data_summary['target'] = model
        summary_final = summary_final.append(summary)
        all_data_final = all_data_final.append(all_data_summary)
    summary_final.to_csv('summary_BOOST_test.csv', index=False, sep = ',')
    all_data_final.to_csv('BOOST_test.csv', index=False, sep = ',')
if __name__ == '__main__':
    main()

