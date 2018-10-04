# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import DialogTeacher
from parlai.core.pytorch_data_teacher import ParlAIDataset
from .build import build

import csv
import random
import os


class DefaultTeacher(DialogTeacher):

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'reddit',
                                       opt['datatype'].split(':')[0] + '.txt')
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path, 'r', newline='') as read:
            csv_read = csv.reader(read, delimiter='\t')
            for line in csv_read:
                line = line[5:7]
                fields = [
                    s.replace('EOS', '\n').replace('START', '').strip()
                    for s in line
                ]
                context = fields[0][-150:]
                if not context: continue
                response = fields[1]
                yield (context, [response], None, [response], None), True


class DefaultDataset(ParlAIDataset):
    pass


