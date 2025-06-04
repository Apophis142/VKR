import argparse


parser = argparse.ArgumentParser(description="Calculating metrics for models")
parser.add_argument("-m", "--model", default="pairframe",
                    choices=["pairframe", "sequenceframe", "none", "target"],
                    help="Type of model to use. Use 'none' to calculate metrics for x. Use 'target' to calculate metrics for y")
parser.add_argument("-c", "-center", "--center-model", default="EnlightenGAN",
                    choices=["EnlightenGAN", "RetinexNet"],
                    help="Model used to process center frame in sequence")
parser.add_argument("-w", "--weights-path", required=True, type=str,
                    help="Path to pretrained model's weight")
parser.add_argument("-fl", "--frames-sequence-length", required=True, type=int,
                    help="Length of frames sequence for model")
parser.add_argument("test-x-paths", type=str,
                    help="Path to test directory with low light frames")
parser.add_argument("test-y-paths", type=str,
                    help="Path to test directory with normal light / processed frames")
parser.add_argument("-k", "--key", required=True, type=str,
                    help="Short key with which will begin the row with results")
parser.add_argument("-resize", "--resize-shape", nargs=2, default=[960, 512], type=int,
                    help="Resize shape for every image model will be trained on. Use it to fasten training, save memory or bring dataset to the same size (default: 960 512)")
parser.add_argument("-dtype", "--tensor-dtype", default="float32", type=str,
                    help="Frames and model's data type")
