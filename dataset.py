import torch


#Data set Format:
# For each batch: get (image, x_p, p) where image in size (b, c, h, w) denotes the ground truth next image, 
# x_p in size (b, c * time_step, h, w) denotes past drawing trejectories, c in size (b, num_class) denotes the class of each sample.
# Maybe also possible to get ground truth next n steps (b, c * n, h, w) as additional element to compute loss on autoregressive steps


class Dataset(torch.utils.data.Dataset):
    # TODO