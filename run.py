
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import NMF  # Non-Negative Matrix Factorization
from sklearn.decomposition import TruncatedSVD

import sys
import numpy as np
from utils import analyse

print('Arguments:', str(sys.argv))

assert len(sys.argv) >= 5, 'Please specify all required arguments when running this script'

dataset_to_use = sys.argv[1].upper()
model_to_run = sys.argv[2].upper()
reducer_to_use = sys.argv[3].upper()
number_of_components_to_consider = [int(v) for v in sys.argv[4:]]

print('Dataset to use:', dataset_to_use)
print('Model to run:', model_to_run)
print('Reducer to use:', reducer_to_use)
print('Number of components to consider:', number_of_components_to_consider)

numTrials = 5
numSplits = 5

files = {
  'dataUNISIM1': {'path': 'dataUNISIM1.txt', 'delimiter': '\t'},
  'dataUNISIM2': {'path': 'dataUNISIM2.txt', 'delimiter': '\t'}
}

models = {
  'GTB': {
    'function': GradientBoostingRegressor,
    'model_params': {'random_state': 0},
    'grid_search_params': {
      'min_samples_split': [0.05, 0.1, 0.2, 0.3],
      'n_estimators': [50, 100, 150],
      'learning_rate': [0.01, 0.1, 0.5],
      'loss': ['ls', 'lad', 'huber']
    }
  },
  'KRR': {
    'function': KernelRidge,
    'model_params': {},
    'grid_search_params': [
      {'kernel': ['poly'], 'degree': [2,3,4], 'alpha': [1e0, 0.1, 1e-2, 1e-3]},
      {'kernel': ['rbf'], 'gamma':  np.logspace(-3, 3, 7), 'alpha': [1e0, 0.1, 1e-2, 1e-3]}  #[1e-3, 1e-1, 1e1]
    ]
  },

 'GPR': {  # GPR has not been used in the paper
    'function': GaussianProcessRegressor,
    'model_params': {},
    'grid_search_params': [
      {'kernel': [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))], 'alpha': np.logspace(-2, 0, 3)},
      {'kernel': [1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)],'alpha': np.logspace(-2, 0, 3)},
      {'kernel': [1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 10.0),
                                periodicity_bounds=(1.0, 10.0))],'alpha': np.logspace(-2, 0, 3)}
    ]
  },

  'MLP': {
    'function': MLPRegressor,
    'model_params': {'max_iter': 400, 'verbose': 0, 'random_state': 0},
    'grid_search_params': {
      'learning_rate': ["invscaling"],
      'learning_rate_init': [0.001, 0.01, 0.1],
      'hidden_layer_sizes': [(25,), (50), (100,), (150,), (50,25), (50,50), (100,50), (100, 100), (150, 100)],
      'activation': ["logistic", "relu", "tanh"]
    }
  },
  'SVR': {
    'function': SVR,
    'model_params': {},
    'grid_search_params': [
      {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [1e1, 1e3, 1e5, 1e7]},
      {'kernel': ['rbf'], 'gamma': np.logspace(-3, 3, 7), 'C': [1e1, 1e3, 1e5, 1e7]}  # [1e-3, 1e-1, 1e1]
    ]
    ## Alternative parameters
    #'grid_search_params': {
    #  'kernel': ['rbf'],
    #  'gamma':  np.logspace(-2, 2, 5), #   [1e-5, 1e-3, 1e-1, 1e1],
    # 'C': [1e0, 1e1, 1e2, 1e3]
  },
  'ENET':{
    'function': ElasticNet,
    'model_params': {'max_iter': 100000, 'random_state': 0},
    'grid_search_params': {
      'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
      'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
    }
  },
  'KNN':{
    'function': KNeighborsRegressor,
    'model_params': {'n_jobs': -1},
    'grid_search_params': {
      'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      'weights': ('uniform', 'distance')
    }
  }
}

def get_reducers(n_components):
  if n_components == 0: return {'none': 0}
  return {
    'PCA': {
      'function': PCA,
      'args': {'n_components': n_components, 'random_state': 0}
    },
    'ISOMAP': {
      'function': Isomap,
      'args': {'n_components': n_components, 'n_jobs': -1}
    },
    'KPCA': {
      'function': KernelPCA,
      'args': {'n_components': n_components, 'random_state': 0, 'n_jobs': -1}
    },
    'LLE': {
      'function': LocallyLinearEmbedding,
      'args': {'n_components': n_components, 'random_state': 0, 'n_jobs': -1}
    },
    'NMF': {
      'function': NMF,
      'args': {'n_components': n_components, 'random_state': 0}
    },
    'TSVD': {
      'function': TruncatedSVD,
      'args': {'n_components': n_components, 'random_state': 0}
    }
  }


for dataset_name, file_info in files.items():
  if dataset_to_use != None and dataset_name.upper() != dataset_to_use:
    continue

  dataset = np.loadtxt(file_info['path'], delimiter=file_info['delimiter'], dtype=np.float64)

  for model_name, model_info in models.items():
    if model_to_run != None and model_name.upper() != model_to_run:
      continue

    for n_components in number_of_components_to_consider:
      reducers = get_reducers(n_components)

      for reducer_name, reducer_info in reducers.items():
        if reducer_to_use != 'NONE' and reducer_name.upper() != reducer_to_use and n_components != 0:
          continue

        if n_components == 0:
          reducer = None
          analysis_name = 'Surrogate_' + dataset_name + '_' + model_name
        else:
          reducer = reducer_info['function'](**reducer_info['args'])
          analysis_name = 'Surrogate_' + dataset_name + '_' + model_name + '_' + reducer_name + '_with_' \
            + str(n_components) + '_' + 'components'

        analyse(
          reducer_name=reducer_name,
          analysis_name=analysis_name,
          dataset=dataset,
          dataset_name=dataset_name,
          model=model_info['function'](**model_info['model_params']),
          grid_search_params=model_info['grid_search_params'],
          reducer=reducer,
          numTrials=numTrials,
          numSplits=numSplits,
          model_name=model_name,
          dimension=n_components,
        )




