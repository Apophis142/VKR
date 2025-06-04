import cv2
import torch
from torchvision import transforms
import numpy as np
from utils import dtypes

import time


from options.fps_measuring_options import parser
args = parser.parse_args()


print(args)

device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cpu")
dtype = dtypes[args.tensor_dtype]

batch_preprocess = transforms.Compose([
    lambda batch_arr: torch.stack([torch.cat([
        torch.tensor(_frame) for _frame in frames
    ], dim=2) for frames in batch_arr], dim=0),
    lambda t: t.transpose(1, 3) / 255,
])
batch_center_preprocess = transforms.Compose([
    lambda batch_arr: torch.stack([torch.tensor(_frame) for _frame in batch_arr], dim=0),
    lambda t: t.transpose(1, 3) / 255,
])


if args.frames_sequence_length == 1:
    if args.center_model == "EnlightenGAN":
        from models.EnligthenGAN import EnlightenOnnxModel

        model = EnlightenOnnxModel()
    elif args.center_model == "RetinexNet":
        from models.RetinexNet import RetinexNet

        model = RetinexNet("models/weights/RetinexNet/").to(dtype).to(device)
    else:
        raise ValueError(f"Unknown center model: {args.center_model}")
elif args.model == "sequenceframe":
    from models.FrameSequenceModel import FrameSequenceModel

    model = FrameSequenceModel(args.frames_sequence_length, args.weights, args.center_model, device, dtype)
elif args.model == "pairframe":
    from models.FramePairModel import FramePairModel

    model = FramePairModel(args.weights, args.center_model, device, dtype)
else:
    raise ValueError("Unknown model: %s" % args.model)


vid = cv2.VideoCapture(args.video_path)
FPS = vid.get(cv2.CAP_PROP_FPS)

batch = [[] for _ in range(args.batch_size)]
batch_center = []
batch_id = 0
batch_frame_id = 0
frame_counter = 0

vid.set(cv2.CAP_PROP_POS_FRAMES, args.number_of_frames_to_skip)

out_frames = []

global_start = time.time()
start = time.time()
timers = []
while True:
    ret, frame =vid.read()
    if not ret:
        break

    if batch_id < args.batch_size:
        batch[batch_id].append(frame)
        batch_frame_id += 1
        if batch_frame_id == args.frames_sequence_length // 2 + 1:
            batch_center.append(frame)
        if batch_frame_id == args.frames_sequence_length:
            batch_frame_id = 0
            batch_id += 1
    else:
        start = time.time()
        tensor_batch = batch_preprocess(batch)
        tensor_batch_center = batch_center_preprocess(batch_center)
        frame_counter += args.batch_size * args.frames_sequence_length

        if args.frames_sequence_length == 1:
            if args.center_model == "EnlightenGAN":
                tensor_batch_center = tensor_batch_center.view(-1, 3, *tensor_batch_center.shape[-2:]).numpy()
                pred = []
                for k in range(tensor_batch_center.shape[0]):
                    pred.append(model.predict(tensor_batch_center[k:k + 1, :, :, :]))
                processed_batch = torch.cat(pred, dim=0)
            else:
                processed_batch = model.predict(tensor_batch_center.to(dtype).to(device)).detach().cpu()
        else:
            processed_batch = model(tensor_batch, tensor_batch_center).view(-1, 3, *tensor_batch.shape[-2:])
        frames_array = np.clip(processed_batch.transpose(1, 3).numpy() * 255, 0, 255).astype(np.uint8)
        out_frames.extend([frames_array[i, :, :, :] for i in range(args.batch_size * args.frames_sequence_length)])
        if args.verbose:
            print("Batch processed: %.4fs. Processed %d/%d frames" %
                  (time.time() - start, frame_counter, args.number_of_frames_to_check))

        batch_id = 0
        batch = [[] for _ in range(args.batch_size)]
        batch_center = []
        timers.append(time.time() - start)
        start = time.time()

    if frame_counter >= args.number_of_frames_to_check:
        break

global_timer = time.time() - global_start
vid.release()

import pickle

import os
if not os.path.exists(args.path_to_file):
    os.makedirs(args.path_to_file)

filename = f"{args.model}x{args.frames_sequence_length}x{args.batch_size}_{args.center_model}.pkl"
with open(args.path_to_file + filename, 'wb+') as file:
    pickle.dump((timers, global_timer), file)
with open(args.path_to_file + 'FPS.txt', 'a') as file:
    print("%s: %.4f" % (f"{args.model}x{args.frames_sequence_length}x{args.batch_size}_{args.center_model}",
                        frame_counter / global_timer), file=file)

if args.save_results:
    print("Saving results into %s" % args.save_results + '.avi')
    out = cv2.VideoWriter(args.save_results + '.avi', -1, FPS,
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for frame in out_frames:
        out.write(frame)
    out.release()

if args.show_results:
    print("Showing results...")
    for frame in out_frames:
        cv2.imshow("processed", frame)
        time.sleep(1 / FPS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
