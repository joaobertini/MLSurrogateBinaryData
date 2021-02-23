
import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score



def r2_adj(observation, prediction):
    r2 = r2_score(observation, prediction)
    (n, p) = observation.shape
    return 1 - (1-r2) * (n-1) / (n-p-1)

def savePlot(filename, x, y, english=True):
    line = [min(y), max(y)]

    plt.clf()

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    fig, ax = plt.subplots()
    if english:
      ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))
      ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))
    else:
      ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x).replace('.', ',')))
      ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.2f}'.format(x).replace('.', ',')))

    ax.scatter(x, y, color='red')
    ax.plot(line, line, color='blue', linewidth=3)

    ax.set_xlabel('Surrogate model estimate (billion USD)' if english else 'Estimativa do modelo auxiliar (US$ bi)', fontsize=15)
    ax.set_ylabel('Simulator output (billion USD)' if english else 'Saída do simulador (US$ bi)', fontsize=15)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)


def saveCsv(filename, prediction, original):
    prediction = pd.DataFrame(prediction, original)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    prediction.to_csv(filename + '.csv', decimal='.', sep=';')


def analyse(analysis_name, dataset, dataset_name, model, grid_search_params, reducer, numTrials, numSplits):
    print()
    print(analysis_name, flush=True)
    print()

    numResults = numTrials * numSplits
    result_MSE = np.zeros(numResults)
    result_R2 = np.zeros(numResults)
    menorErro = 0

    time_spent_training = []
    time_spent_testing = []

    count = 0
    for i in range(numTrials):
        k_fold = ShuffleSplit(n_splits=numSplits, random_state=i)

        for train_indices, test_indices in k_fold.split(dataset):
            print("Iteracao %d de %d:" % (count + 1, numResults), flush=True)
            sys.stdout.flush()

            # Train start
            train_start = datetime.datetime.now()

            trainOriginalFeatures = dataset[train_indices, :-1]
            trainOriginalLabels = dataset[train_indices, -1].reshape(-1, 1)

            if reducer is None:
              trainReducedFeatures = trainOriginalFeatures
            else:
              reducerWeights = reducer['fit'](trainOriginalFeatures)
              trainReducedFeatures = reducer['transform'](trainOriginalFeatures, reducerWeights)

            reducedFeaturesScaler = StandardScaler().fit(trainReducedFeatures)
            trainScaledReducedFeatures = reducedFeaturesScaler.transform(trainReducedFeatures)

            labelsScaler = MinMaxScaler().fit(trainOriginalLabels)
            trainScaledLabels = labelsScaler.transform(trainOriginalLabels)

            selector = GridSearchCV(model, grid_search_params, cv=5, n_jobs=-1, verbose=0)
            selector.fit(trainScaledReducedFeatures, trainScaledLabels.ravel())

            #Train end
            train_end = datetime.datetime.now()
            time_spent_training.append((train_end - train_start).total_seconds())

            # Test start
            test_start = datetime.datetime.now()

            testOriginalFeatures = dataset[test_indices, :-1]
            testOriginalLabels = dataset[test_indices, -1].reshape(-1, 1)

            if reducer is None:
              testReducedFeatures = testOriginalFeatures
            else:
              testReducedFeatures = reducer['transform'](testOriginalFeatures, reducerWeights)

            testScaledReducedFeatures = reducedFeaturesScaler.transform(testReducedFeatures)

            scaledPredictedLabels = selector.predict(testScaledReducedFeatures).reshape(-1, 1)
            predictedLabels = labelsScaler.inverse_transform(scaledPredictedLabels)

            result_MSE[count] = np.sqrt(mean_squared_error(testOriginalLabels, predictedLabels))
            result_R2[count] = r2_adj(testOriginalLabels, predictedLabels.reshape(-1, 1))

            # Test end
            test_end = datetime.datetime.now()
            time_spent_testing.append((test_end - test_start).total_seconds())

            if count == 0:
                menorErro = result_MSE[count]
                bestSelector = selector
                bestReducer = reducer
                bestReducerWeights = reducerWeights
                bestReducedFeaturesScaler = reducedFeaturesScaler
                bestLabelsScaler = labelsScaler
                print("- Novo menor erro=%f" % menorErro)
            else:
                if result_MSE[count] < menorErro:
                    menorErro = result_MSE[count]
                    bestSelector = selector
                    bestReducer = reducer
                    bestReducerWeights = reducerWeights
                    bestReducedFeaturesScaler = reducedFeaturesScaler
                    bestLabelsScaler = labelsScaler
                    print("- Novo menor erro=%f" % menorErro)
                else:
                    print("- Nada alterado")

            count += 1

    print("RMSE media %2.5f std %2.5f " % (result_MSE.mean(), result_MSE.std()))
    print("R2 media %2.5f std %2.5f " % (result_R2.mean(), result_R2.std()))

    features = dataset[:, :-1]
    labels = dataset[:, -1].reshape(-1, 1)


    if reducer is None:
      reducedFeatures = features
    else:
      reducedFeatures = bestReducer['transform'](features, bestReducerWeights)

    scaledReducedFeatures = bestReducedFeaturesScaler.transform(reducedFeatures)

    scaledPrediction = bestSelector.predict(scaledReducedFeatures).reshape(-1, 1)
    prediction = bestLabelsScaler.inverse_transform(scaledPrediction)

    rMSE = np.sqrt(mean_squared_error(prediction, labels))
    rR2 = r2_adj(prediction.reshape(-1, 1), labels)
    print("RMSE %2.5f R2 %2.5f " % (rMSE, rR2), flush=True)

    saveCsv(
      filename='output/csvs/' + dataset_name + '/' + analysis_name,
      prediction=prediction,
      original=labels.ravel()
    )
    savePlot(
      filename='output/figures/english/' + dataset_name + '/' + analysis_name,
      x=prediction,
      y=labels.ravel(),
      english=True
    )
    savePlot(
      filename='output/figures/portuguese/' + dataset_name + '/' + analysis_name,
      x=prediction,
      y=labels.ravel(),
      english=False
    )

    print()
    print("Training time: avg %.3fs std %.3fs" % (np.mean(time_spent_training), np.std(time_spent_training)), flush=True)
    print("Testing time: avg %.3fs std %.3fs" % (np.mean(time_spent_testing), np.std(time_spent_testing)), flush=True)

