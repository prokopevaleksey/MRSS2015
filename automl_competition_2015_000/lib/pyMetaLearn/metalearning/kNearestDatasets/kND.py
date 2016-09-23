from collections import OrderedDict
import itertools

import sys
import types

import numpy as np
import pandas as pd

import sklearn.ensemble
import sklearn.utils

import pyMetaLearn.metalearning.create_datasets as create_datasets
from pyMetaLearn.metalearning.meta_base import Run


class KNearestDatasets(object):
    def __init__(self, distance='l1', random_state=None, distance_kwargs=None):
        self.distance = distance
        self.model = None
        self.distance_kwargs = distance_kwargs
        self.metafeatures = None
        self.runs = None
        self.best_hyperparameters_per_dataset = None
        self.random_state = sklearn.utils.check_random_state(random_state)

        if self.distance_kwargs is None:
            self.distance_kwargs = {}

    def fit(self, metafeatures, runs):
        # Metafeatures is a dataframe with one row for each dataset
        assert isinstance(metafeatures, pd.DataFrame)
        assert metafeatures.values.dtype == np.float64
        assert np.isfinite(metafeatures.values).all()
        # Runs is a dictionary with one entry for every dataset
        assert isinstance(runs, dict)
        assert len(runs) == metafeatures.shape[0], \
            (len(runs), metafeatures.shape[0])

        self.metafeatures = metafeatures
        self.runs = runs

        def cmp_runs(x, y):
            if np.isnan(x.result):
                return 1  # x is "larger"
            elif np.isnan(y.result):
                return -1  # x is "smaller"
            else:
                return cmp(x.result, y.result)  # compare numbers

        # for each dataset, sort the runs according to their result
        best_hyperparameters_per_dataset = {}
        for dataset_name in runs:
            best_hyperparameters_per_dataset[dataset_name] = \
                sorted(runs[dataset_name], cmp=cmp_runs)
        self.best_hyperparameters_per_dataset = best_hyperparameters_per_dataset

        if self.distance == 'learned':
            self.distance_kwargs['random_state'] = self.random_state
            sys.stderr.write("Going to use the following RF hyperparameters\n")
            sys.stderr.write(str(self.distance_kwargs) + "\n")
            sys.stderr.flush()
            self.model = LearnedDistanceRF(**self.distance_kwargs)
            return self.model.fit(metafeatures, runs)
            """
            elif self.distance == 'mfs_l1':
                # This implements metafeature selection as described by Matthias
                # Reif in 'Metalearning for evolutionary parameter optimization
                # of classifiers'
                self.model = MetaFeatureSelection(**self.distance_kwargs)
                return self.model.fit(metafeatures, runs)
            elif self.distance == 'mfw_l1':
                self.model = MetaFeatureSelection(mode='weight', **self.distance_kwargs)
                return self.model.fit(metafeatures, runs)
            """
        elif self.distance not in ['l1', 'l2', 'random']:
            raise NotImplementedError(self.distance)

    def kNearestDatasets(self, x, k=1):
        # Actually, these are no longer datasets but tasks
        # k=-1 return all datasets sorted by distance
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        distances = self._calculate_distances_to(x)
        sorted_distances = sorted(distances.items(), key=lambda t: t[1])
        # sys.stderr.write(str(sorted_distances))
        # sys.stderr.write("\n")
        # sys.stderr.flush()

        if k == -1:
            k = len(sorted_distances)
        return sorted_distances[:k]

    def kBestSuggestions(self, x, k=1, exclude_double_configurations=True):
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        sorted_distances = self.kNearestDatasets(x, -1)
        kbest = []

        if exclude_double_configurations:
            added_configurations = set()
            for dataset_name, distance in sorted_distances:
                best_configurations = self.best_hyperparameters_per_dataset[
                    dataset_name]

                if len(best_configurations) == 0:
                    continue

                best_configuration = best_configurations[0]
                if best_configuration not in added_configurations:
                    added_configurations.add(best_configuration)
                    kbest.append((dataset_name, distance, best_configuration))

                if k != -1 and len(kbest) >= k:
                    break
        else:
            for dataset_name, distance in sorted_distances:
                best_configurations = self.best_hyperparameters_per_dataset[
                    dataset_name]

                if len(best_configurations) == 0:
                    continue

                best_configuration = best_configurations[0]
                kbest.append((dataset_name, distance, best_configuration))

        if k == -1:
            k = len(kbest)
        return kbest[:k]

    def _calculate_distances_to(self, other):
        distances = {}
        assert isinstance(other, pd.Series)
        assert other.values.dtype == np.float64
        if not np.isfinite(other.values).all():
            raise ValueError("%s contains non-finite metafeatures" % str(other))

        if other.name in self.metafeatures.index:
            raise ValueError("You are trying to calculate the distance to a "
                             "dataset which is in your base data.")

        if self.distance in ['l1', 'l2', 'mfs_l1', 'mfw_l1']:
            metafeatures, other = self._scale(self.metafeatures, other)
        else:
            metafeatures = self.metafeatures

        for idx, mf in metafeatures.iterrows():
            dist = self._calculate_distance(mf, other)
            distances[mf.name] = dist

        return distances

    def _scale(self, metafeatures, other):
        assert isinstance(other, pd.Series)
        assert other.values.dtype == np.float64
        scaled_metafeatures = metafeatures.copy(deep=True)
        other = other.copy(deep=True)

        mins = scaled_metafeatures.min()
        maxs = scaled_metafeatures.max()
        # I also need to scale the target dataset meta features...
        mins = pd.DataFrame(data=[mins, other]).min()
        maxs = pd.DataFrame(data=[maxs, other]).max()
        scaled_metafeatures = (scaled_metafeatures - mins) / (maxs - mins)
        other = (other -mins) / (maxs - mins)
        return scaled_metafeatures, other

    def _calculate_distance(self, d1, d2):
        distance_fn = getattr(self, "_" + self.distance)
        return distance_fn(d1, d2)

    def _l1(self, d1, d2):
        """http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Taxicab_norm_or_Manhattan_norm"""
        return np.sum(abs(d1 - d2))

    def _l2(self, d1, d2):
        """http://en.wikipedia.org/wiki/Norm_%28mathematics%29#p-norm"""
        return np.sqrt(np.sum((d1 - d2)**2))

    def _random(self, d1, d2):
        return self.random_state.random_sample()

    def _learned(self, d1, d2):
        model = self.model
        x = np.hstack((d1, d2))

        predictions = model.predict(x)
        # Predictions are between -1 and 1, -1 indicating a negative correlation.
        # Since we evaluate the dataset with the smallest distance, we would
        # evaluate the dataset with the most negative correlation
        #logger.info(predictions)
        #logger.info(predictions[0] * -1)
        return (predictions[0] * -1) + 1

    def _mfs_l1(self, d1, d2):
        d1 = d1.copy() * self.model.weights
        d2 = d2.copy() * self.model.weights
        return self._l1(d1, d2)

    def _mfw_l1(self, d1, d2):
        return self._mfs_l1(d1, d2)


class LearnedDistanceRF(object):
    # TODO: instead of a random forest, the user could provide a generic
    # import call with which it is possible to import a class which
    # implements the sklearn fit and predict function...
    def __init__(self, n_estimators=100, max_features=0.2,
                 min_samples_split=2, min_samples_leaf=1, n_jobs=1,
                 random_state=None, oob_score=False):
        if isinstance(random_state, str):
            random_state = int(random_state)
        rs = sklearn.utils.check_random_state(random_state)
        rf = sklearn.ensemble.RandomForestRegressor(
            n_estimators=int(n_estimators), max_features=float(max_features),
            min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf),
            criterion="mse", random_state=rs, oob_score=oob_score, n_jobs=int(n_jobs))
        self.model = rf

    def fit(self, metafeatures, runs):
        X, Y = self._create_dataset(metafeatures, runs)
        model = self._fit(X, Y)
        return model

    def _create_dataset(self, metafeatures, runs):
        runs = self._apply_surrogates(metafeatures, runs)
        X, Y = create_datasets.create_predict_spearman_rank(
                metafeatures, runs, "permutation")
        return X, Y

    def _fit(self, X, Y):
        self.model.fit(X, Y)
        return self.model

    def _apply_surrogates(self, metafeatures, runs, n_estimators=500,
                          n_jobs=1, random_state=None, oob_score=True):

        # Find out all configurations for which we need to know result in
        # order to calculate the correlation coefficient
        configurations = set()
        configurations_per_run = dict()
        outcomes_per_run = dict()
        for name in runs:
            configurations_per_run[name] = set()
            outcomes_per_run[name] = dict()
            for experiment in runs[name]:
                # TODO: refactor the classes so params are hashable
                configurations.add(str(experiment.configuration))
                configurations_per_run[name].add(str(experiment.configuration))
                outcomes_per_run[name][str(experiment.configuration)] = \
                    experiment.result

        filled_runs = {}
        for name in runs:
            print ".",
            run = runs[name]
            # Transfer all previous experiments
            filled_runs[name] = run

            train_x = []
            train_y = []
            predict = []

            for configuration in configurations:
                param = eval(configuration)
                if configuration in configurations_per_run[name]:
                    train_x.append(param)
                    train_y.append(outcomes_per_run[name][configuration])
                else:
                    predict.append(param)

            train_x = pd.DataFrame(train_x)
            train_y = pd.Series(train_y)
            predict = pd.DataFrame(predict)

            # Hacky procedure to be able to use the scaling/onehotencoding on
            #  all data at the same time
            stacked = train_x.append(predict, ignore_index=True)
            stacked_y = train_y.append(pd.Series(np.zeros((len(predict)))))

            if len(predict) == 0:
                continue

            if isinstance(random_state, str):
                random_state = int(random_state)
            rs = sklearn.utils.check_random_state(random_state)
            rf = sklearn.ensemble.RandomForestRegressor(
                n_estimators=int(n_estimators),
                criterion="mse", random_state=rs, oob_score=oob_score,
                n_jobs=int(n_jobs))

            # For the y array we have to convert the NaNs already here; maybe
            #  it would even better to leave them out...
            stacked_y.fillna(1, inplace=True)

            stacked, stacked_y = _convert_pandas_to_npy(stacked, stacked_y)
            num_training_samples = len(train_x)
            train_x = stacked[:num_training_samples]
            predict_x = stacked[num_training_samples:]
            train_y = stacked_y[:num_training_samples]

            rf = rf.fit(train_x, train_y)

            prediction = rf.predict(predict_x)
            for x_, y_ in itertools.izip(predict.iterrows(), prediction):
                # Remove all values which are nan
                params = {pair[0]: pair[1] for pair in x_[1].to_dict().items()
                          if pair[1] == pair[1]}
                params = OrderedDict([pair for pair in sorted(params.items())])
                # TODO: add a time prediction surrogate
                filled_runs[name].append(Run(params, y_, 1))
            filled_runs[name].sort()

        return filled_runs

    def predict(self, metafeatures):
        assert isinstance(metafeatures, np.ndarray)
        return self.model.predict(metafeatures)


"""
class MetaFeatureSelection(object):
    def __init__(self, max_number_of_combinations=10, random_state=None,
                 k=1, max_features=0.5, mode='select'):
        self.max_number_of_combinations = max_number_of_combinations
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.k = k
        self.max_features = max_features
        self.weights = None
        self.mode = mode

    def fit(self, metafeatures, runs):
        self.datasets = metafeatures.index
        self.all_other_datasets = {}  # For faster indexing
        self.all_other_runs = {}  # For faster indexing
        self.parameter_distances = defaultdict(dict)
        self.best_hyperparameters_per_dataset = {}
        self.mf_names = metafeatures.columns
        self.kND = KNearestDatasets(distance='l1')

        for dataset in self.datasets:
            self.all_other_datasets[dataset] = \
                pd.Index([name for name in self.datasets if name != dataset])

        for dataset in self.datasets:
            self.all_other_runs[dataset] = \
                {key: runs[key] for key in runs if key != dataset}

        for dataset in self.datasets:
            self.best_hyperparameters_per_dataset[dataset] = \
                sorted(runs[dataset], key=lambda t: t.result)[0]

        for d1, d2 in itertools.combinations(self.datasets, 2):
            hps1 = self.best_hyperparameters_per_dataset[d1]
            hps2 = self.best_hyperparameters_per_dataset[d2]
            keys = set(hps1.params.keys())
            keys.update(hps2.params.keys())
            dist = 0
            for key in keys:
                # TODO: test this; it can happen that string etc occur
                try:
                    p1 = float(hps1.params.get_value(key, 0))
                    p2 = float(hps2.params.get_value(key, 0))
                    dist += abs(p1 - p2)
                except:
                    dist += 0 if hps1.params.get_value(key, 0) == \
                                 hps2.params.get_value(key, 0) else 1

                    # dist += abs(hps1.params.get_value(key, 0) - hps2.params.get_value(key, 0))
            self.parameter_distances[d1][d2] = dist
            self.parameter_distances[d2][d1] = dist

        if self.mode == 'select':
            self.weights = self._fit_binary_weights(metafeatures)
        elif self.mode == 'weight':
            self.weights = self._fit_weights(metafeatures)

        sys.stderr.write(str(self.weights))
        sys.stderr.write('\n')
        sys.stderr.flush()
        return self.weights

    def _fit_binary_weights(self, metafeatures):
        best_selection = None
        best_distance = sys.maxint

        for i in range(2,
                       int(np.round(len(self.mf_names) * self.max_features))):
            sys.stderr.write(str(i))
            sys.stderr.write('\n')
            sys.stderr.flush()

            combinations = []
            for j in range(self.max_number_of_combinations):
                combination = []
                target = i
                maximum = len(self.mf_names)
                while len(combination) < target:
                    random = self.random_state.randint(maximum)
                    name = self.mf_names[random]
                    if name not in combination:
                        combination.append(name)

                combinations.append(pd.Index(combination))

            for j, combination in enumerate(combinations):
                dist = 0
                for dataset in self.datasets:
                    hps = self.best_hyperparameters_per_dataset[dataset]
                    self.kND.fit(metafeatures.loc[self.all_other_datasets[
                                                      dataset], combination],
                                 self.all_other_runs[dataset])
                    nearest_datasets = self.kND.kBestSuggestions(
                        metafeatures.loc[dataset, np.array(combination)],
                        self.k)
                    for nd in nearest_datasets:
                        # print "HPS", hps.params, "nd", nd[2]
                        dist += self.parameter_distances[dataset][nd[0]]

                if dist < best_distance:
                    best_distance = dist
                    best_selection = combination

        weights = dict()
        for metafeature in metafeatures:
            if metafeature in best_selection:
                weights[metafeature] = 1
            else:
                weights[metafeature] = 0
        return pd.Series(weights)

    def _fit_weights(self, metafeatures):
        best_weights = None
        best_distance = sys.maxint

        def objective(weights):
            dist = 0
            for dataset in self.datasets:
                self.kND.fit(metafeatures.loc[self.all_other_datasets[
                    dataset], :] * weights, self.all_other_runs[dataset])
                nearest_datasets = self.kND.kBestSuggestions(
                    metafeatures.loc[dataset, :] * weights, self.k)
                for nd in nearest_datasets:
                    dist += self.parameter_distances[dataset][nd[0]]

            return dist

        for i in range(10):
            w0 = np.ones((len(self.mf_names, ))) * 0.5 + \
                 (np.random.random(size=len(self.mf_names)) - 0.5) * i / 10
            bounds = [(0, 1) for idx in range(len(self.mf_names))]

            res = scipy.optimize.minimize \
                (objective, w0, bounds=bounds, method='L-BFGS-B',
                 options={'disp': True})

            if res.fun < best_distance:
                best_distance = res.fun
                best_weights = pd.Series(res.x, index=self.mf_names)

        return best_weights
"""


def _convert_pandas_to_npy(X, Y, replace_missing_with=0,
                           scaling=None):
    """Nominal values are replaced with a one hot encoding and missing
     values represented with zero."""

    if replace_missing_with != 0:
        raise NotImplementedError(replace_missing_with)

    num_fields = 0
    attribute_arrays = []
    keys = []

    for idx, attribute in enumerate(X.iteritems()):
        # iteritems lazily iterates over (index, value) pairs
        attribute_name = attribute[0].lower()
        attribute_type = attribute[1].dtype
        row = attribute[1]

        if attribute_type in (np.float64, np.int64):
            try:
                rval = _parse_numeric(row, scaling=scaling)
            except Exception as e:
                print idx, attribute_name
                raise e
            if rval is not None:
                keys.append(attribute_name)
                attribute_arrays.append(rval)
                num_fields += 1

        elif attribute_type == 'object':
            rval = _parse_nominal(row)
            if rval is not None:
                attribute_arrays.append(rval)
                num_fields += rval.shape[1]
                if rval.shape[1] == 1:
                    keys.append(attribute_name)
                else:
                    vals = [attribute_name + ":" + str(possible_value) for
                            possible_value in range(rval.shape[1])]
                    keys.extend(vals)

        else:
            raise NotImplementedError((attribute_name, attribute_type))

    dataset_array = np.ndarray((X.shape[0], num_fields))

    col_idx = 0
    for attribute_array in attribute_arrays:
        length = attribute_array.shape[1]
        dataset_array[:, col_idx:col_idx + length] = attribute_array
        col_idx += length

    if Y.dtype == 'object':
        encoding = _encode_labels(Y)
        Y = np.array([encoding[value] for value in Y], np.int32)
    elif Y.dtype == np.float64:
        Y = np.array([value for value in Y], dtype=np.float64)
    Y = Y.reshape((-1, 1)).ravel()

    return dataset_array, Y

    # ###########################################################################
    # Helper functions for converting attributes


def _convert_attribute_type(attribute_type):
    # Input looks like:
    # {'?','GB','GK','GS','TN','ZA','ZF','ZH','ZM','ZS'}
    # real
    # etc...

    if isinstance(attribute_type, types.StringTypes):
        attribute_type = attribute_type.lower()
    elif isinstance(attribute_type, list):
        attribute_type = "nominal"
    else:
        raise NotImplementedError()

    # a string indicates something like real, integer while nominal
    # is represented as an array
    if attribute_type in ("real", "integer", "numeric"):
        dtype = np.float64
    elif attribute_type == "nominal":
        dtype = 'object'
    else:
        print attribute_type
        import sys

        sys.stdout.flush()
        raise NotImplementedError()

    return dtype


def _parse_nominal(row):
    # This few lines perform a OneHotEncoding, where missing
    # values represented by none of the attributes being active (
    # a feature which i could not implement with sklearn).
    # Different imputation strategies can easily be added by
    # extracting a method from the else clause.
    # Caution: this methodology only keeps values that are
    # encountered in the dataset. If this is a subset of the
    # possible values of the arff file, only the subset is
    # encoded via the OneHotEncoding
    encoding = _encode_labels(row)

    if len(encoding) == 0:
        return None

    array = np.zeros((row.shape[0], len(encoding)))

    for row_idx, value in enumerate(row):
        # The value can actually be nan, for example if build an array from
        # numerous dicts, that's why we check for equality with itself
        if row[row_idx] is not None and row[row_idx] == row[row_idx]:
            array[row_idx][encoding[row[row_idx]]] = 1

    return array


def _normalize_scaling(array):
    # Apply scaling here so that if we are setting missing values
    # to zero, they are still zero afterwards
    X_min = np.nanmin(array, axis=0)
    X_max = np.nanmax(array, axis=0)
    # Numerical stability...
    if (X_max - X_min) > 0.0000000001:
        array = (array - X_min) / (X_max - X_min)

    return array


def _normalize_standardize(array):
    raise NotImplementedError()
    mean = np.nanmean(array, axis=0, dtype=np.float64)
    X = array - mean
    std = np.nanstd(X, axis=0, dtype=np.float64)
    return X / std


def _parse_numeric(row, scaling=None):
    # NaN and None will be treated as missing values
    array = np.array(row).reshape((-1, 1))

    if not np.any(np.isfinite(array)):
        return None

    if scaling is None:
        pass
    elif scaling == "scale":
        array = _normalize_scaling(array)
    elif scaling == "zero_mean":
        array = _normalize_standardize(array)
    else:
        raise NotImplementedError(str(scaling))
    fixed_array = np.ma.fix_invalid(array, copy=True, fill_value=0)

    if not np.isfinite(fixed_array).all():
        print fixed_array
        raise NotImplementedError()

    return fixed_array


def _encode_labels(row):
    discrete_values = set(row)
    discrete_values.discard(None)
    discrete_values.discard(np.NaN)
    # Adds reproduceability over multiple systems
    discrete_values = sorted(discrete_values)
    encoding = OrderedDict()
    for row_idx, possible_value in enumerate(discrete_values):
        encoding[possible_value] = row_idx
    return encoding

