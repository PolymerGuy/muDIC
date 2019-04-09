from abc import abstractmethod, ABCMeta


class FieldInterpolator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def Nn(self, e, n):
        raise NotImplementedError

    @abstractmethod
    def dxNn(self, e, n):
        raise NotImplementedError

    @abstractmethod
    def dyNn(self, e, n):
        raise NotImplementedError
