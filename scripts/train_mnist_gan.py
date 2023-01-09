import os
import sys
sys.path.insert(0, './')
import argparse
import numpy as np

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.models.mnist import MNIST_FC_D, MNIST_FC_G
from src.datasets import mnist_loader

def imshow(image, dimshuffle = None):
    '''
    image of shape [height, width, channels]
    '''
    if dimshuffle != None:
        image = np.transpose(image, dimshuffle)

    image = np.clip(image, a_min = 0.0, a_max = 1.0)
    channel_num = image.shape[-1]
    if channel_num == 3:
        plt.imshow(image)
    elif channel_num == 1:
        stacked_image = np.concatenate([image, image, image], axis = 2)
        plt.imshow(stacked_image)
    else:
        raise ValueError('image format is wrong, channel_num = %d'%channel_num)

def plot_mnist_data(true_rows, fake_rows, cols, data, netG, input_dim, use_gpu, output_file):

    rows = true_rows + fake_rows
    cols = cols

    plt.figure(figsize = (rows, cols))
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace = 0., hspace = 0.)

    data_list = [data_batch for data_batch, label_batch in data]
    true_data_idx = 0

    for row_idx in range(rows):

        for col_idx in range(cols):

            if row_idx < true_rows:
                image = np.array(data_list[true_data_idx][0])
                true_data_idx += 1
            else:
                noise = torch.randn(1, input_dim)
                if use_gpu:
                    noise = noise.cuda()
                image = netG(noise).detach()
                image = np.array(image[0])
            image = image.reshape(28, 28, 1)

            image = (image + 1.) / 2.

            subplot_idx = row_idx * cols + col_idx 
            plt.subplot(gs1[subplot_idx])
            imshow(image)
            plt.xticks([])
            plt.yticks([])

    # plt.suptitle('First %d Rows are Real Data, Last %d Rows are Fake Data'%(true_rows, fake_rows))
    plt.savefig(output_file, ddi = 500, bbox_inches = 'tight')
    plt.clf()

def main(args):

    netG = MNIST_FC_G()
    netD = MNIST_FC_D()
    loss_func = nn.BCELoss()

    train_loader, test_loader = mnist_loader(batch_size = args.batch_size, batch_size_test = 1)

    use_cuda = True if torch.cuda.is_available() and args.gpu != 'cpu' else False
    device = 'cuda:0' if use_cuda else 'cpu'

    if use_cuda:
        netD = netD.cuda()
        netG = netG.cuda()
        loss_func = loss_func.cuda()

    optimG = torch.optim.Adam(netG.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    optimD = torch.optim.Adam(netD.parameters(), lr = 1e-3, betas = (0.9, 0.99))

    true_target = torch.ones([args.batch_size, 1], device = device)
    fake_target = torch.zeros([args.batch_size, 1], device = device)

    for epoch in range(args.n_epochs):

        print('Epoch %d'%epoch)

        for idx, (true_data, _) in enumerate(train_loader):

            if true_data.shape[0] != args.batch_size:           # Incomplete batch
                continue

            if use_cuda:
                true_data = true_data.cuda()

            # Update Generator
            optimG.zero_grad()

            z = torch.randn([args.batch_size, args.nz], device = device)
            fake_data = netG(z)
            fake_loss = loss_func(netD(fake_data), true_target)

            fake_loss.backward()
            optimG.step()

            # Update discriminator
            optimD.zero_grad()

            true_loss = loss_func(netD(true_data), true_target)
            fake_loss = loss_func(netD(fake_data.detach()), fake_target)
            total_loss = (true_loss + fake_loss) / 2

            total_loss.backward()
            optimD.step()

        if (epoch + 1) % 10 == 0:
            netG.eval()
            plot_mnist_data(true_rows = 2, fake_rows = 8, cols = 10, data = test_loader, netG = netG,
                input_dim = args.nz, use_gpu = use_cuda, output_file = os.path.join(args.outf, '%s_%d.pdf'%(args.model_name, epoch + 1)))
            netG.train()

        torch.save(netG.state_dict(), os.path.join(args.outf, '%s_netG.ckpt'%args.model_name))
        torch.save(netD.state_dict(), os.path.join(args.outf, '%s_netD.ckpt'%args.model_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nz', type = int, default = 20,
        help = 'the number of input dimension, default = 20')
    parser.add_argument('--n_epochs', type = int, default = 200,
        help = 'the number of total epochs, default = 200')
    parser.add_argument('--batch_size', type = int, default = 128,
        help = 'the size of minibatch, default = 128')
    parser.add_argument('--gpu', type = str, default = '0',
        help = 'the index of GPU to use, default = "0"')
    parser.add_argument('--outf', type = str, default = None,
        help = 'the output directory, default = None')
    parser.add_argument('--model_name', type = str, default = None,
        help = 'the name of the model, default = None')

    args = parser.parse_args()
    if args.gpu != None and args.gpu != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print('Use GPU %s'%args.gpu)
    else:
        print('Use default GPU' if args.gpu == None else 'Use CPUs')

    if args.outf == None:
        raise ValueError('The output folder cannot be None.')
    if args.model_name == None:
        raise ValueError('The model name cannot be None.')

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    print('**** Arguments ****')
    for arg, value in args._get_kwargs():
        print(arg + ':', value)
    print('*******************')

    main(args)
