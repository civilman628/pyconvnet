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

from data import *
import numpy.random as nr
import numpy as n
import random as r

#numpixel = 150528

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        #self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 256
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            #d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            #d['data1'] = n.require((d['data1'] - n.mean(d['data1'], axis=0,dtype=n.single)[n.newaxis,:]) / n.std(d['data1'], axis=0,dtype=n.single)[n.newaxis,:], dtype=n.single, requirements='C')
            #d['data2'] = n.require((d['data2'] - n.mean(d['data2'], axis=0,dtype=n.single)[n.newaxis,:]) / n.std(d['data2'], axis=0,dtype=n.single)[n.newaxis,:], dtype=n.single, requirements='C')
            #d['data3'] = n.require((d['data3'] - n.mean(d['data3'], axis=0,dtype=n.single)[n.newaxis,:]) / n.std(d['data3'], axis=0,dtype=n.single)[n.newaxis,:], dtype=n.single, requirements='C')
            #d['data4'] = n.require((d['data4'] - n.mean(d['data4'], axis=0,dtype=n.single)[n.newaxis,:]) / n.std(d['data4'], axis=0,dtype=n.single)[n.newaxis,:], dtype=n.single, requirements='C')
            
            d['data1'] = n.require(d['data1']-n.mean(d['data1'], axis=0,dtype=n.single), dtype=n.single, requirements='C') #-n.mean(d['data1'], axis=0,dtype=n.single)
            d['data2'] = n.require(d['data2']-n.mean(d['data2'], axis=0,dtype=n.single), dtype=n.single, requirements='C')
            d['data3'] = n.require(d['data3']-n.mean(d['data3'], axis=0,dtype=n.single), dtype=n.single, requirements='C')
            d['data4'] = n.require(d['data4']-n.mean(d['data4'], axis=0,dtype=n.single), dtype=n.single, requirements='C')
 
            #d['labels'] = n.require(d['labels'].reshape((1, d['data1'].shape[1])), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'], dtype=n.single, requirements='C')

    def get_next_batch(self): 
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self) 
        
        d['data1'] = n.require(d['data1']-n.mean(d['data1'], axis=0,dtype=n.single), dtype=n.single, requirements='C') #-n.mean(d['data1'], axis=0,dtype=n.single)
        d['data2'] = n.require(d['data2']-n.mean(d['data2'], axis=0,dtype=n.single), dtype=n.single, requirements='C')
        d['data3'] = n.require(d['data3']-n.mean(d['data3'], axis=0,dtype=n.single), dtype=n.single, requirements='C')
        d['data4'] = n.require(d['data4']-n.mean(d['data4'], axis=0,dtype=n.single), dtype=n.single, requirements='C')
 
        d['labels'] = n.require(d['labels'], dtype=n.single, requirements='C')

        #return epoch, batchnum, [datadic['data1'], datadic['data2'],datadic['labels']]
        return epoch, batchnum, [datadic['data1'], datadic['data2'],datadic['data3'], datadic['data4'],datadic['labels']]
        #return epoch, batchnum, [datadic['data1'],datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx != 4 else 1
        #return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        #return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
        return n.require((data).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 256 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 10
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        self.batches_generated = 0

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data4'].shape[1]*self.data_mult), dtype=n.single)]

        #cropped1 = self.cropped_data[0].copy()
        #cropped2 = self.cropped_data[0].copy()
        #cropped3 = self.cropped_data[0].copy()
        cropped4 = self.cropped_data[0].copy()
        cropped5 = self.cropped_data[0].copy()
        #cropped = self.cropped_data[0].copy()
        
        #self.__trim_borders(n.require(self.data_dic[0]['data1'],dtype=n.single, requirements='C') , cropped1)
        #self.__trim_borders(n.require(self.data_dic[0]['data2'],dtype=n.single, requirements='C') , cropped2)
        #self.__trim_borders(n.require(self.data_dic[0]['data3'],dtype=n.single, requirements='C') , cropped3)
        self.__trim_borders(n.require(self.data_dic[0]['data4'],dtype=n.single, requirements='C') , cropped4)
        self.__trim_borders(n.require(self.data_dic[0]['data5'],dtype=n.single, requirements='C') , cropped5)
        #self.__trim_borders2(n.require(self.data_dic[0]['data4'],dtype=n.single, requirements='C'), n.require(self.data_dic[0]['data5'],dtype=n.single, requirements='C'), cropped4, cropped5)
        labels = n.require(self.data_dic[0]['labels'], dtype=n.single, requirements='C').copy()

        
        #testimages=cropped1.copy()
        testimages=cropped4.copy()

        #cropped1=n.require(cropped1-n.mean(cropped1, axis=0,dtype=n.single), dtype=n.single, requirements='C')
        #cropped2=n.require(cropped2-n.mean(cropped2, axis=0,dtype=n.single), dtype=n.single, requirements='C')
        #cropped3=n.require(cropped3-n.mean(cropped3, axis=0,dtype=n.single), dtype=n.single, requirements='C')
        cropped4=n.require(cropped4-n.mean(cropped4, axis=0,dtype=n.single), dtype=n.single, requirements='C')
        cropped5=n.require(cropped5-n.mean(cropped5, axis=0,dtype=n.single), dtype=n.single, requirements='C')

        #cropped=n.require(cropped-n.mean(cropped, axis=0,dtype=n.single), dtype=n.single, requirements='C')

        #return epoch, batchnum, [cropped1,cropped2,cropped3,cropped4,labels],testimages
        return epoch, batchnum, [cropped4, cropped5, labels], testimages
        #return epoch, batchnum, [cropped4, labels], testimages
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx != 2 else 1
        #return self.inner_size**2 * 3 if idx == 0 else 1
        #return self.inner_size**2 * 3 if idx != 4 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        #return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
        return n.require((data).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        y = x.reshape(3, 256, 256, x.shape[1])
        for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

        #if self.test: # don't need to loop over cases
        #    if self.multiview:
        #        start_positions = [(0,0),  (0, self.border_size*2),
        #                           (self.border_size, self.border_size),
        #                          (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
        #        end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
        #        for i in xrange(self.num_views/2):
        #            pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
        #            target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
        #            target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
        #    else:
        #        pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
        #        target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))

        #else:
        #    for c in xrange(x.shape[1]): # loop over cases
        #        startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
        #        endY, endX = startY + self.inner_size, startX + self.inner_size
        #        pic = y[:,startY:endY,startX:endX, c]
        #        if nr.randint(2) == 0: # also flip the image with 50% probability
        #            pic = pic[:,:,::-1]
        #        target[:,c] = pic.reshape((self.get_data_dims(),))

    def __trim_borders2(self, x1, x2, target1, target2):
        y1 = x1.reshape(3, 256, 256, x1.shape[1])
        y2 = x2.reshape(3, 256, 256, x2.shape[1])
        for c in xrange(x1.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic1 = y1[:,startY:endY,startX:endX, c]
                pic2 = y2[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic1 = pic1[:,:,::-1]
                    pic2 = pic2[:,:,::-1]
                target1[:,c] = pic1.reshape((self.get_data_dims(),))
                target2[:,c] = pic2.reshape((self.get_data_dims(),))

        #if self.test: # don't need to loop over cases
        #    if self.multiview:
        #        start_positions = [(0,0),  (0, self.border_size*2),
        #                           (self.border_size, self.border_size),
        #                          (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
        #        end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
        #        for i in xrange(self.num_views/2):
        #            pic1 = y1[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
        #            target1[:,i * x1.shape[1]:(i+1)* x1.shape[1]] = pic1.reshape((self.get_data_dims(),x1.shape[1]))
        #            target1[:,(self.num_views/2 + i) * x1.shape[1]:(self.num_views/2 +i+1)* x1.shape[1]] = pic1[:,:,::-1,:].reshape((self.get_data_dims(),x1.shape[1]))
        #            pic2 = y2[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
        #            target2[:,i * x2.shape[1]:(i+1)* x2.shape[1]] = pic.reshape((self.get_data_dims(),x2.shape[1]))
        #            target2[:,(self.num_views/2 + i) * x2.shape[1]:(self.num_views/2 +i+1)* x2.shape[1]] = pic2[:,:,::-1,:].reshape((self.get_data_dims(),x2.shape[1]))
        #    else:
        #        pic1 = y1[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
        #        target1[:,:] = pic1.reshape((self.get_data_dims(), x1.shape[1]))
        #        pic2 = y2[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
        #        target2[:,:] = pic2.reshape((self.get_data_dims(), x2.shape[1]))
        #else:
        #    for c in xrange(x1.shape[1]): # loop over cases
        #        startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
        #        endY, endX = startY + self.inner_size, startX + self.inner_size
        #        pic1 = y1[:,startY:endY,startX:endX, c]
        #        pic2 = y2[:,startY:endY,startX:endX, c]
        #        if nr.randint(2) == 0: # also flip the image with 50% probability
        #            pic1 = pic1[:,:,::-1]
        #            pic2 = pic2[:,:,::-1]
        #        target1[:,c] = pic1.reshape((self.get_data_dims(),))
        #        target2[:,c] = pic2.reshape((self.get_data_dims(),))