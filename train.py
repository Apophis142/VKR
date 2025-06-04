from utils import dtypes, loss_functions
import torch


from options.training_options import parser
args = parser.parse_args()


if args.multi_threading_training:
    from utils.multi_thread_training import train_nn
else:
    from utils.one_thread_training import train_nn

print(args)
if args.model == "pairframe":
    from models.FramePairModel import OneFramePairProcessing
    from data.batch_loader import preload_all_pair_frames_paths as load_paths

    net = OneFramePairProcessing()
elif args.model == "sequenceframe":
    from models.FrameSequenceModel import OneSequenceProcessing
    from data.batch_loader import preload_all_sequence_frames_paths as load_paths

    net = OneSequenceProcessing(frame_sequence_length=args.frames_sequence_length)
else:
    raise ValueError


train_dataset = load_paths(
    args.train_x_paths,
    args.train_y_paths,
    args.frames_sequence_length,
)
test_dataset = load_paths(
    args.test_x_paths,
    args.test_y_paths,
    args.frames_sequence_length,
)
if args.gpu_id == -1:
    device = torch.device("cpu")
elif args.gpu_id >= 0 and isinstance(args.gpu_id, int):
    device = torch.device("cuda:%d" % args.gpu_id)
else:
    raise Exception
net = net.to(dtypes[args.tensor_dtype])
net = net.to(device)
hist = train_nn(
    net,
    train_dataset,
    test_dataset,
    learning_rate=args.learning_rate,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    device=device,
    loss_function=loss_functions[args.loss_function],
    filename_to_save=args.filename_to_save,
    epoch_frequency_save=args.epoch_save_frequency,
    resize=args.resize_shape,
    dtype=dtypes[args.tensor_dtype]
)
