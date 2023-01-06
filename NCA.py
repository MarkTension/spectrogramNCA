import torch

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])


def perchannel_conv(x, filters):
    '''filters: [filter_n, h, w]'''
    b, ch, h, w = x.shape
    y = x.reshape(b*ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


def perception(x):
    filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
    return perchannel_conv(x, filters)


def get_living_mask(x):
  alpha = x[:, 3:4, :, :] # used to be on last dim
  return torch.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1




class CA(torch.nn.Module):
    """ the cellular automata class.
    """

    def __init__(self, chn=12, hidden_n=96, living_mask=False):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn*4, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.living_mask = living_mask

    def forward(self, x, update_rate=0.5):
        """_summary_

        Args:
            x (tensor): current state
            update_rate (float, optional): chance of an update per cell. (stochastic cell update) Defaults to 0.5.

        Returns:
            _type_: _description_
        """      
        y = perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape

        udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
        
        if self.living_mask:
            udpate_mask = udpate_mask * get_living_mask(x)
        # post_mask = get_living_mask(y)  

        return x + y * udpate_mask

    def seed(self, n:int, sz_w=256, sz_h=256):
        """initiates the sample pool with black pixel state.

        Args:
            n (int): number of pools
            sz_w (int, optional): the width dimension. Defaults to 256
            sz_h (int, optional): the height dimension. Defaults to 256

        Returns:
            torch tensor: initialized sample pool
        """        

        seed = torch.zeros(n, self.chn, sz_w, sz_h)
        seed[:, 3:, sz_w//2, sz_h//2] = 1.0
        return seed
