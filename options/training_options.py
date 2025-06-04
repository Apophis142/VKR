import argparse


parser = argparse.ArgumentParser(description="Training model to color correcting videos")
parser.add_argument("-m", "--model", default="pairframe", choices=["pairframe", "sequenceframe"],
                    help="Type of model to be trained")
parser.add_argument("train-x-paths", type=str,
                    help="Path to train directory with low light frames")
parser.add_argument("train-y-paths", type=str,
                    help="Path to train directory with normal light / processed frames")
parser.add_argument("test-x-paths", type=str,
                    help="Path to test directory with low light frames")
parser.add_argument("test-y-paths", type=str,
                    help="Path to test directory with normal light / processed frames")
parser.add_argument("-fl", "--frames-sequence-length", type=int, required=True,
                    help="Length of frames sequence for model")
parser.add_argument("-b", "--batch-size", default=1, type=int,
                    help="Batch size for training (default: 1)")
parser.add_argument("-lr", "--learning-rate", default=.0001, type=float,
                    help="Learning rate (default: 1e-4)")
parser.add_argument("-loss", "--loss_function", default="mae", choices=["mae", "mse"],
                    help="Loss function for training (default: mae)")
parser.add_argument("-resize", "--resize-shape", nargs=2, default=[960, 512], type=int,
                    help="Resize shape for every image model will be trained on. Use it to fasten training, save memory or bring dataset to the same size (default: 960 512)")
parser.add_argument("-dtype", "--tensor_dtype", default="float",
                    choices=["double", "float", "float32", "float16", "half"],
                    help="Number of bytes to store one number with floating point. Not recommended to use 'float16' (default: float)")
parser.add_argument("-epoch", "--num-epochs", default=100, type=int,
                    help="Number of epochs model will be training for (default: 100)")
parser.add_argument("-gpu", "--gpu-id", default=0, type=int,
                    help="GPU's id to use. Use -1 for CPU (default: 0)")
parser.add_argument("-sf", "--epoch_save_frequency", default=10, type=int,
                    help="Frequency of interim results saving (default: 10)")
parser.add_argument("-save", "--filename-to-save", required=True, type=str,
                    help="Filename or path to save results")
parser.add_argument("-multi", "--multi_threading_training", action=argparse.BooleanOptionalAction,
                    help="Use this argument to parallelize preloading batches from drive and optimizing model's parameters. May lead to unstable training process")
