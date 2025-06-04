import cv2
import torch
from torchvision import transforms
import numpy as np
import tqdm

from utils import dtypes

from options.metrics_options import parser
args = parser.parse_args()


from basicsr.metrics import niqe
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


print(args)

img_load = transforms.Compose([
    cv2.imread,
    torch.tensor,
    lambda t: t.to(dtypes[args.tensor_dtype]).transpose(2, 0) / 255,
    transforms.Resize(args.resize_shape)
])
if args.model == "pairframe":
    from models.FramePairModel import FramePairModel
    from data.batch_loader import preload_all_pair_frames_paths, load_batch_pair_frames, \
        preload_all_sequence_frames_paths

    model = FramePairModel(args.weights_path, args.center_model)
    dataset = preload_all_pair_frames_paths(
        args.test_x_paths,
        args.test_y_paths,
        args.frames_sequence_length,
    )

    batch_loader = lambda path: load_batch_pair_frames(path, img_load)
elif args.model == "sequenceframe":
    from models.FrameSequenceModel import FrameSequenceModel
    from data.batch_loader import preload_all_sequence_frames_paths, load_batch_sequence_frames

    model = FrameSequenceModel(args.frames_sequence_length, args.weights_path, args.center_model)
    dataset = preload_all_sequence_frames_paths(
        args.test_x_paths,
        args.test_y_paths,
        args.frames_sequence_length,
    )

    batch_loader = lambda path: load_batch_sequence_frames(path, img_load)
elif args.model == "none" or args.model == "target":
    from data.batch_loader import preload_all_frames_paths, xy_batch_loader

    model = lambda t: t
    dataset = preload_all_frames_paths(
        args.test_x_paths,
        args.test_y_paths,
    )
    batch_loader = lambda path: xy_batch_loader(path, img_load)
else:
    raise ValueError

with tqdm.tqdm(total=len(dataset)) as pbar:
    metric_niqe = 0
    metric_ssim = 0
    metric_pnsr = 0
    counter = 0

    for paths in dataset:
        if args.model in ("pairframe", "target", "none"):
            x, y = batch_loader([paths])
            xs = [x[0]]
            ys = [y[0]]
        elif args.model == "sequenceframe":
            x, y = batch_loader([paths])
            xs = [x[0][j*3:(j+1)*3, :, :] for j in range(1, args.frames_sequence_length+1)]
            ys = [y[0][j*3:(j+1)*3, :, :] for j in range(args.frames_sequence_length)]
        if args.model == "pairframe":
            processed_images = [model(x[:, 3:, :, :], x[:, :3, :, :]).squeeze()]
        elif args.model == "sequenceframe":
            processed_images = model(x[0]).view(-1, 3, *args.resize_shape)
            processed_images = [processed_images[j].squeeze() for j in range(args.frames_sequence_length)]
        elif args.model == "none":
            processed_images = xs
        elif args.model == "target":
            processed_images = ys
        for x, y, processed_image in zip(xs, ys, processed_images):
            x = (x.numpy() * 255).astype(np.uint8)
            y = (y.numpy() * 255).astype(np.uint8)
            processed_image = (processed_image.detach().cpu().numpy() * 255).astype(np.uint8)
            metric_niqe += niqe.calculate_niqe(processed_image, crop_border=0, input_order="CHW")
            metric_ssim += structural_similarity(processed_image, y, channel_axis=0)
            metric_pnsr += peak_signal_noise_ratio(y, processed_image, data_range=255)
            counter += 1

        pbar.update()


import os


if not os.path.exists("metrics"):
    os.makedirs("metrics")

with open("metrics/niqe.txt", 'a') as file:
    print("%s %d frame sequence: %.6f" % (args.key, args.frames_sequence_length, metric_niqe / counter), file=file)
with open("metrics/ssim.txt", 'a') as file:
    print("%s %d frame sequence: %.6f" % (args.key, args.frames_sequence_length, metric_ssim / counter), file=file)
with open("metrics/pnsr.txt", 'a') as file:
    print("%s %d frame sequence: %.6f" % (args.key, args.frames_sequence_length, metric_pnsr / counter), file=file)
