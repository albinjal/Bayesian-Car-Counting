import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
from matplotlib import cm, pyplot as plt
from scipy.ndimage import zoom

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # if no file with pth suffix, then load from tar
    if not os.path.exists(os.path.join(args.save_dir, 'best_model.pth')):
        # find tar name
        name = ''
        for file in os.listdir(args.save_dir):
            if file.endswith('.tar'):
                name = file
                break
        checkpoint = torch.load(os.path.join(args.save_dir, name), device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []
    i = 0
    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(i, name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)

            plot_density_map = True
            # Plot density map
            if plot_density_map:
                dm = outputs.squeeze().detach().cpu().numpy()
                dm_normalized = dm / np.max(dm)
                i += 1
                # create a plot to hold the original image and the density map

                img = inputs.cpu().numpy()[0]
                # shape of img is 3xHxW, need to transpose to HxWx3 for printing
                img = np.transpose(img, (1, 2, 0))
                # shift the image to [0, 255]
                img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
                img = img.astype(np.uint8)
                side_by_side = False
                if side_by_side:
                    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                    ax[0].imshow(img)
                    ax[1].imshow(dm_normalized, cmap=cm.jet, vmin=0, vmax=1)
                else:
                    # overlay the density map on the original image

                    # We have to calculate the scale for each dimension.
                    scale_y = img.shape[0]/dm_normalized.shape[0]
                    scale_x = img.shape[1]/dm_normalized.shape[1]

                    # zoom() function will resize/rescale your density map.
                    dm_rescaled = zoom(dm_normalized, (scale_y, scale_x))
                    fig, ax = plt.subplots(figsize=(20, 10))
                    ax.imshow(img)
                    ax.imshow(dm_rescaled, cmap=cm.jet, alpha=0.3, vmin=0, vmax=1)
                # save the figure
                fig.savefig(os.path.join(args.save_dir, name[0] + '.png'))

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
