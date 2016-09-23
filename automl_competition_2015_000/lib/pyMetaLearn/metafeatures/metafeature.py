from abc import ABCMeta, abstractmethod
from StringIO import StringIO
import time
import types

import arff


class AbstractMetaFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _calculate(cls, X, Y, categorical):
        pass

    def __call__(self, X, Y, categorical=None):
        if categorical is None:
            categorical = [False for i in range(X.shape[1])]
        starttime = time.time()
        value = self._calculate(X, Y, categorical)
        endtime = time.time()
        return MetaFeatureValue(self.__class__.__name__, self.type_,
                                0, 0, value, endtime-starttime)


class MetaFeature(AbstractMetaFeature):
    def __init__(self):
        self.type_ = "METAFEATURE"


class HelperFunction(AbstractMetaFeature):
    def __init__(self):
        self.type_ = "HELPERFUNCTION"


class MetaFeatureValue(object):
    def __init__(self, name, type_, fold, repeat, value, time):
        self.name = name
        self.type_ = type_
        self.fold = fold
        self.repeat = repeat
        self.value = value
        self.time = time

    def to_arff_row(self):
        if self.type_ == "METAFEATURE":
            value = self.value
        else:
            value = "?"

        return [self.name, self.type_, self.fold,
                self.repeat, value, self.time]

    def __repr__(self):
        repr = "%s (type: %s, fold: %d, repeat: %d, value: %s, time: %3.3f)"
        repr = repr % tuple(self.to_arff_row()[:4] + [self.to_arff_row()[4]]
                            + [self.to_arff_row()[5]])
        return repr

class DatasetMetafeatures(object):
    def __init__(self, dataset_name, metafeature_values):
        self.dataset_name = dataset_name
        self.metafeature_values = metafeature_values

    def _get_arff(self):
        output = dict()
        output['relation'] = "metafeatures_%s" % (self.dataset_name)
        output['description'] = ""
        output['attributes'] = [('name', 'STRING'),
                                ('type', 'STRING'),
                                ('fold', 'NUMERIC'),
                                ('repeat', 'NUMERIC'),
                                ('value', 'NUMERIC'),
                                ('time', 'NUMERIC')]
        output['data'] = []

        for value in self.metafeature_values:
            output['data'].append(value.to_arff_row())
        return output

    def dumps(self):
        return self._get_arff()

    def dump(self, path_or_filehandle):
        output = self._get_arff()

        if isinstance(path_or_filehandle, types.StringTypes):
            with open(path_or_filehandle, "w") as fh:
                arff.dump(output, fh)
        else:
            arff.dump(output, path_or_filehandle)

    @classmethod
    def load(cls, path_or_filehandle):

        if isinstance(path_or_filehandle, types.StringTypes):
            with open(path_or_filehandle) as fh:
                input = arff.load(fh)
        else:
            input = arff.load(path_or_filehandle)

        dataset_name = input['relation'].replace('metafeatures_', '')
        metafeature_values = []
        for item in input['data']:
            mf = MetaFeatureValue(*item)
            metafeature_values.append(mf)

        return cls(dataset_name, metafeature_values)

    def __repr__(self):
        repr = StringIO()
        repr.write("Metafeatures for dataset %s\n" % self.dataset_name)
        for metefeature_value in self.metafeature_values:
            repr.write("  %s\n" % str(metefeature_value))
        return repr.getvalue()
