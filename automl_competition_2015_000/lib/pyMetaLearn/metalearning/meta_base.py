import logging
import os
import cPickle

import arff
import pandas as pd
import yaml

from ..metafeatures.metafeature import DatasetMetafeatures
from HPOlibConfigSpace.configuration_space import ConfigurationSpace, Configuration


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("META_BASE")


class Run(object):
    def __init__(self, configuration, result, runtime):
        self.configuration = configuration
        self.result = result
        self.runtime = runtime

    def __repr__(self):
        return "Run:\nresult: %3.3f\nruntime: %3.3f\n%s" % \
               (self.result, self.runtime, str(self.configuration))

class Instance(object):
    def __init__(self, name, features):
        self.name = name
        self.features = features

class MetaBase(object):
    def __init__(self, configuration_space, datasets, experiments):
        """Container for dataset metadata and experiment results.

        Constructor arguments:
        - The configuration space
        - instances: A list of instance files
        - experiments: A list in which every entry corresponds to one entry
            in datasets. It must contain a list of Runs.
        """

        self.configuration_space = configuration_space
        self.datasets = dict()
        self.metafeatures = dict()
        self.run_history = dict()

        # Open the yaml files
        self.datasets_list_filename = datasets
        with open(self.datasets_list_filename) as fh:
            datasets = yaml.safe_load(fh)
        self.experiment_list_filename = experiments
        with open(self.experiment_list_filename) as fh:
            experiments = yaml.safe_load(fh)

        for dataset in datasets:
            self.datasets[dataset['name']] = dataset
            metafeature_filename = dataset['metafeature_file']

            metafeatures, possible_metafeature_filenames = \
                self._load_metafeature_file(metafeature_filename)
            if metafeatures is None:
                raise Exception("Could not load metafeatures for dataset %s, "
                                "searched at the following locations %s" %
                                (dataset['name'], possible_metafeature_filenames))

            self.metafeatures[dataset['name']] = metafeatures

        for experiment_list in experiments:
            dataset_name = experiment_list['name']
            runs = []
            for experiment_file in experiment_list['experiments']:
                # experiment_list can either store
                # * HPOlib pickles or a dictionary with the following keys:
                # configuration, result, duration

                if isinstance(experiment_file, dict):
                    runs_ = [self.get_run_from_dict(experiment_file)]

                else:
                    possible_experiment_filenames, runs_ = \
                        self.load_experiment_pkl(experiment_file)
                    if runs_ is None:
                        raise Exception(
                            "Could not load experiments for dataset %s, "
                            "searched at the following locations %s" %
                            (dataset['name'], possible_experiment_filenames))

                runs.extend(runs_)

            if dataset_name not in self.datasets:
                raise ValueError("Experiment list contains experiment for "
                                 "dataset %s which is not in the dataset "
                                 "list." % dataset_name)
            self.run_history[dataset_name] = runs

    def _load_metafeature_file(self, metafeature_filename):
        # If the metafeature filename is a relative path it is useful to
        # check if the file is relative to the datasets file.
        possible_metafeature_filenames = \
            [metafeature_filename,
             os.path.join(os.path.dirname(self.datasets_list_filename),
                          metafeature_filename)]
        metafeatures = None
        for pmf in possible_metafeature_filenames:
            try:
                with open(pmf) as fh:
                    metafeatures = DatasetMetafeatures.load(fh)
                break
            except IOError, OSError:
                pass
        return metafeatures, possible_metafeature_filenames

    def load_experiment_pkl(self, experiment_file):
        possible_experiment_filenames = \
            [experiment_file,
             os.path.join(os.path.dirname(
                 self.experiment_list_filename), experiment_file)]
        runs_ = None
        for pef in possible_experiment_filenames:
            try:
                with open(pef) as fh:
                    if pef.endswith(".pkl"):
                        runs_ = self.read_experiment_pickle(fh)
                    elif pef.endswith(".yaml"):
                        runs_ = self.read_experiment_yaml(fh)
                    else:
                        raise ValueError()

            except IOError, OSError:
                pass
        return possible_experiment_filenames, runs_

    def add_dataset(self, name, file, metafeature_file):
        """Add a new dataset to the meta_base.

        This method should be used when metalearning will be conducted for a
        new dataset for which we do not have any previous experiments."""

        possible_metafeature_filenames, mf = self._load_metafeature_file(
            metafeature_file)
        if mf is None:
            raise Exception("Could not load metafeatures for dataset %s, "
                            "searched at the following locations %s" %
                            (mf, possible_metafeature_filenames))

        self.add_dataset_with_metafeatures(name, file, mf)

    def add_dataset_with_metafeatures(self, name, file, metafeatures):

        # Add the dataset after loading the metafeatures to make sure that we
        #  have a consistent state!
        dct = {'name': name,
               'file': file,
               'metafeature_file': None}
        self.datasets[name] = dct
        # TODO test that metafeatures are added here!
        self.metafeatures[name] = metafeatures

    def get_dataset(self, name):
        """Return dataset attribute"""
        return self.datasets[name]

    def get_datasets(self):
        """Return datasets attribute."""
        return self.datasets

    def get_runs(self, dataset_name):
        """Return a list of all runs for a dataset."""
        return self.run_history[dataset_name]

    def get_all_runs(self):
        """Return a dictionary with a list of all runs"""
        return self.run_history

    def get_metafeatures(self, dataset_name):
        dataset_metafeatures = self.metafeatures[dataset_name]
        mf = {value.name: value.value for value in
              dataset_metafeatures.metafeature_values if
              value.type_ == 'METAFEATURE'}

        metafeatures = pd.Series(data=mf, name=dataset_name)
        return metafeatures

    def get_metafeatures_times(self, dataset_name):
        # TODO abstract this!
        metafeatures_file = self.datasets[dataset_name]['metafeature_file']
        with open(metafeatures_file) as fh:
            metafeatures_arff = arff.load(fh)

        mf = {value[0]: value[5] for value in metafeatures_arff['data']}

        metafeatures = pd.Series(data=mf, name=metafeatures_arff['relation'])
        return metafeatures

    def get_all_metafeatures(self):
        """Create a pandas DataFrame for the metadata of all datasets."""
        series = []

        for key in self.datasets:
            series.append(self.get_metafeatures(key))

        retval = pd.DataFrame(series)
        return retval

    def get_all_metafeatures_times(self):
        """Create a pandas DataFrame for the metadata of all datasets."""
        series = []

        for key in self.datasets:
            series.append(self.get_metafeatures_times(key))

        retval = pd.DataFrame(series)
        return retval

    def read_experiment_pickle(self, fh):
        runs = list()
        trials = cPickle.load(fh)
        for trial in trials["trials"]:
            params = trial['params']
            for key in params:
                try:
                    params[key] = float(params[key])
                except:
                    pass

            configuration = Configuration(self.configuration_space, **params)
            runs.append(Run(configuration, trial["result"], trial["duration"]))

        return runs

    def get_run_from_dict(self, dct):
        configuration = Configuration(self.configuration_space, **dct['configuration'])
        return Run(configuration, dct['result'], dct['duration'])


