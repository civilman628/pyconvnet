# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as n
from numpy.random import randn, rand, random_integers
import os
from util import *

BATCH_META_FILE = "labels.txt"
TotalClasses = 100

class DataProvider:
    BATCH_REGEX = re.compile('^data_batch_(\d+)(\.\d+)?$')
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        if batch_range == None:
            batch_range = DataProvider.get_batch_nums(data_dir)
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]

        self.data_dir = data_dir
        self.batch_range = batch_range
        self.curr_epoch = init_epoch
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
        self.batch_meta = self.get_batch_meta(data_dir)
        self.data_dic = None
        self.test = test
        self.batch_idx = batch_range.index(init_batchnum)
        self.trainindex=0
        self.testindex=0

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        return epoch, batchnum, self.data_dic
    
    def __add_subbatch(self, batch_num, sub_batchnum, batch_dic):
        subbatch_path = "%s.%d" % (os.path.join(self.data_dir, self.get_data_file_name(batch_num)), sub_batchnum)
        if os.path.exists(subbatch_path):
            sub_dic = unpickle(subbatch_path)
            self._join_batches(batch_dic, sub_dic)
        else:
            raise IndexError("Sub-batch %d.%d does not exist in %s" % (batch_num,sub_batchnum, self.data_dir))
        
    def _join_batches(self, main_batch, sub_batch):
        main_batch['data'] = n.r_[main_batch['data'], sub_batch['data']]
        
    def get_batch(self, batch_num):
        if os.path.exists(self.get_data_file_name(batch_num) + '.1'): # batch in sub-batches
            dic = unpickle(self.get_data_file_name(batch_num) + '.1')
            sb_idx = 2
            while True:
                try:
                    self.__add_subbatch(batch_num, sb_idx, dic)
                    sb_idx += 1
                except IndexError:
                    break
        else:
            dic = unpickle(self.get_data_file_name(batch_num))
        return dic
    
    def get_data_dims(self):
        return self.batch_meta['num_vis']
    
    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        self.curr_batchnum = self.batch_range[self.batch_idx] # ???  why num=2
        if self.batch_idx == 0: # we wrapped
            self.curr_epoch += 1
            
    def get_next_batch_idx(self):
        return (self.batch_idx + 1) % len(self.batch_range)
    
    def get_next_batch_num(self):
        return self.batch_range[self.get_next_batch_idx()]
    
    # get filename of current batch
    def get_data_file_name(self, batchnum=None):
        if batchnum is None:
            batchnum = self.curr_batchnum
        return os.path.join(self.data_dir, 'data_batch_%d.mat' % batchnum)
    
    @classmethod
    def get_instance(cls, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, type="default", dp_params={}, test=False):
        # why the fuck can't i reference DataProvider in the original definition?
        #cls.dp_classes['default'] = DataProvider
        type = type or DataProvider.get_batch_meta(data_dir)['dp_type'] # allow data to decide data provider
        if type.startswith("dummy-"):
            name = "-".join(type.split('-')[:-1]) + "-n"
            if name not in dp_types:
                raise DataProviderException("No such data provider: %s" % type)
            _class = dp_classes[name]
            dims = int(type.split('-')[-1])
            return _class(dims)
        elif type in dp_types:
            _class = dp_classes[type]
            return _class(data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
        raise DataProviderException("No such data provider: %s" % type)
    
    @classmethod
    def register_data_provider(cls, name, desc, _class):
        if name in dp_types:
            raise DataProviderException("Data provider %s already registered" % name)
        dp_types[name] = desc
        dp_classes[name] = _class
        
    @staticmethod
    def get_batch_meta(data_dir):
        return  [line.strip() for line in open(os.path.join(data_dir, BATCH_META_FILE), 'r')]
        #return sio.loadmat(os.path.join(data_dir, BATCH_META_FILE), chars_as_strings=True)
    
    @staticmethod
    def get_batch_filenames(srcdir):
        return sorted([f for f in os.listdir(srcdir) if DataProvider.BATCH_REGEX.match(f)], key=alphanum_key)
    
    @staticmethod
    def get_batch_nums(srcdir):
        names = DataProvider.get_batch_filenames(srcdir)
        return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))
        
    @staticmethod
    def get_num_batches(srcdir):
        return len(DataProvider.get_batch_nums(srcdir))

class MemoryDataProvider(DataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_dic = []
        for i in self.batch_range:
            self.data_dic += [self.get_batch(i)]
    
    def get_next_batch(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()

        return epoch, batchnum, self.data_dic[batchnum - self.batch_range[0]]

class LabeledDataProvider(DataProvider):   
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_num_classes(self):
        #return len(self.batch_meta['label_names'])
        return TotalClasses
    
class LabeledMemoryDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_dic = []
            
    def get_next_batch(self):  
        if self.test == False:
            i= self.batch_range[self.trainindex]
            self.data_dic = [unpickle(self.get_data_file_name(i))]
            self.trainindex+=1

            if self.trainindex ==len(self.batch_range):
               self.trainindex =0
            #print "\n train batch number: %d\n" %i
        else:
            i= self.batch_range[self.testindex]
            #i=349
            self.data_dic = [unpickle(self.get_data_file_name(i))]
            self.testindex+=1
            if self.testindex ==len(self.batch_range):
               self.testindex =0
            print "\n test batch number: %d\n" %i

        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        bidx = batchnum - self.batch_range[0]
        return epoch, batchnum, self.data_dic[0] # 1st bidx=0, 2nd = 1,
    
dp_types = {"default": "The default data provider; loads one batch into memory at a time",
            "memory": "Loads the entire dataset into memory",
            "labeled": "Returns data and labels (used by classifiers)",
            "labeled-memory": "Combination labeled + memory",
            "dummy-n": "Dummy data provider for n-dimensional data",
            "dummy-labeled-n": "Labeled dummy data provider for n-dimensional data"}
dp_classes = {"default": DataProvider,
              "memory": MemoryDataProvider,
              "labeled": LabeledDataProvider,
              "labeled-memory": LabeledMemoryDataProvider,
              }
    
class DataProviderException(Exception):
    pass
