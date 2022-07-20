# -*- coding: utf-8 -*-
from __future__ import print_function, division
from loss.cls.ce import CrossEntropyLoss, SigmoidCELoss
from loss.cls.l1 import L1Loss
from loss.cls.nll import NLLLoss
from loss.cls.mse import MSELoss

PyMICClsLossDict = {"CrossEntropyLoss": CrossEntropyLoss,
    "SigmoidCELoss": SigmoidCELoss,
    "L1Loss":  L1Loss,
    "MSELoss": MSELoss,
    "NLLLoss": NLLLoss}
