from abc import abstractmethod, ABC
from typing import List
from tensornetwork import Node

class Sketcher(ABC):

    def __init__(self, m, d, D, N, K):
        """
        Takes in sketcher parameters
        :param m: The sketch dimension size
        :param d: The physical dimension
        :param D: The bond dimension
        :param N: The amount of tensors in the MPO
        :param K: The amount of MPOs to embedd
        """
        self.m = m
        self.d = d
        self.D = D
        self.N = N
        self.K = K

    @abstractmethod
    def sketch(self, mpos: List[List[Node]]):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
