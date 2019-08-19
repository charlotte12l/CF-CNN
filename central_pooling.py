import numpy as np
import math
import torch
import torch.nn as nn


TABEL=np.load('lookup_table.npy')


   
class central_pooling(nn.Module):
    def __init__(self, inputsize):
        super(central_pooling, self).__init__()
        self.inputsize = inputsize
        self.n1, self.n2, self.n3 = self.get_n(self.inputsize)

    def forward(self, x):
        out = []
        # raw central pooling
        n3_up = x[:, :, :3 * math.ceil(self.n3 / 2), :]
        n3_row_max = nn.MaxPool2d((3, 1), stride=(3, 1))
        n3_up_out1 = n3_row_max(n3_up)
        out.append(n3_up_out1)

        n2_up = x[:, :, 3 * math.ceil(self.n3 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2), :]
        n2_row_max = nn.MaxPool2d((2, 1), stride=(2, 1))
        n2_up_out1 = n2_row_max(n2_up)
        out.append(n2_up_out1)

        n1_up = x[:, :,
                3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(
                    self.n1 / 2), :]
        # eliminate (1,1) maxpooling
        out.append(n1_up)

        index = 3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(self.n1 / 2)
        if ((self.n1 - math.ceil(self.n1 / 2)) > 0):
            n1_bottom = x[:, :, index:int(index + (self.n1 - math.ceil(self.n1 / 2))), :]
            index = int(index + (self.n1 - math.ceil(self.n1 / 2)))
            # eliminate (1,1) maxpooling
            out.append(n1_bottom)

        if ((self.n2 - math.ceil(self.n2 / 2)) > 0):
            n2_bottom = x[:, : , index:int(index + 2 * (self.n2 - math.ceil(self.n2 / 2))), :]
            index = int(index + 2 * (self.n2 - math.ceil(self.n2 / 2)))
            n2_bottom_out = n2_row_max(n2_bottom)
            out.append(n2_bottom_out)
        if ((self.n3 - math.ceil(self.n3 / 2)) > 0):
            n3_bottom = x[:, :, index:int(index + 3 * (self.n3 - math.ceil(self.n3 / 2))), :]
            index = int(index + 3 * (self.n3 - math.ceil(self.n3 / 2)))
            n3_bottom_out = n3_row_max(n3_bottom)
            out.append(n3_bottom_out)

        concat = torch.cat(out, dim=2)

        # column central pooling
        out1 = []

        n3_left = concat[:, :, :, :3 * math.ceil(self.n3 / 2)]
        n3_col_max = nn.MaxPool2d((1, 3), stride=(1, 3))
        n3_left_out = n3_col_max(n3_left)
        out1.append(n3_left_out)

        n2_left = concat[:, :, :, 3 * math.ceil(self.n3 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2)]
        n2_col_max = nn.MaxPool2d((1, 2), stride=(1, 2))
        n2_left_out = n2_col_max(n2_left)
        out1.append(n2_left_out)

        n1_left = concat[:, :, :,
                  3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2):3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(
                      self.n1 / 2)]
        # eliminate (1,1) maxpooling
        out1.append(n1_left)

        index = 3 * math.ceil(self.n3 / 2) + 2 * math.ceil(self.n2 / 2) + math.ceil(self.n1 / 2)
        if ((self.n1 - math.ceil(self.n1 / 2)) > 0):
            n1_right = concat[:, :, :, index:int(index + (self.n1 - math.ceil(self.n1 / 2)))]
            index = int(index + (self.n1 - math.ceil(self.n1 / 2)))
            # eliminate (1,1) maxpooling
            out1.append(n1_right)
        if ((self.n2 - math.ceil(self.n2 / 2)) > 0):
            n2_right = concat[:, :, :, index:int(index + 2 * (self.n2 - math.ceil(self.n2 / 2)))]
            index = int(index + 2 * (self.n2 - math.ceil(self.n2 / 2)))
            n2_right_out = n2_col_max(n2_right)
            out1.append(n2_right_out)
        if ((self.n3 - math.ceil(self.n3 / 2)) > 0):
            n3_right = concat[:, :, :, index:int(index + 3 * (self.n3 - math.ceil(self.n3 / 2)))]
            index = int(index + 3 * (self.n3 - math.ceil(self.n3 / 2)))
            n3_right_out = n3_col_max(n3_right)
            out1.append(n3_right_out)

        concat1 = torch.cat(out1, dim=3)
        return concat1

    def get_n(self, input_size):
        n1=math.floor(input_size/8)
        n2=math.floor(input_size/4)
        n3=math.floor(input_size/8)
        residual=input_size-n1*1-n2*2-n3*3
        L=self.look_up(residual)

        n1=n1+L[1]
        n2=n2+L[2]
        n3=n3+L[3]
        assert(n1+2*n2+3*n3==input_size)

        return n1, n2, n3

    def look_up(self, r):
        return TABEL[:,r]

    
        
if __name__ == '__main__':
    a=np.random.rand(3,9,9)[np.newaxis,:,:,:]

    a=torch.from_numpy(a)
    print(a.size())
    #print(a)
    cnn = central_pooling(9)
    output = cnn(a)
    print(output.size())
   # print(output)

    


