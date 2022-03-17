#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

import re
import random
import itertools
import numpy as np
import pickle as pk
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict, Counter

from datetime import datetime
from dateutil.relativedelta import relativedelta

from konlpy.tag import Komoran
komoran = Komoran()

from newsutil import *
newspath = NewsPath()





if __name__ == '__main__':
    ## Filenames
    