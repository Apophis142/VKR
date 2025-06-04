import argparse


parser = argparse.ArgumentParser(description="Measuring FPS, running video processing")
parser.add_argument('-m', '--model', type=str, required=True, choices=["pairframe", "sequenceframe", "None"],
                    help="Type of model to use")
parser.add_argument('-fl', '--frames_sequence_length', type=int, required=True,
                    help="Length of frames sequence for model")
parser.add_argument('-w', '--weights', type=str, required=True,
                    help="Path to model's weights")
parser.add_argument('-c', '-center', '--center-model', type=str, required=True,
                    choices=["EnlightenGAN", "RetinexNet"],
                    help="Model used to process center frame in sequence")
parser.add_argument('-vid', '--video-path', type=str, required=True,
                    help="Path to video for processing")
parser.add_argument('-skip', '-skip-frames', '--number-of-frames-to-skip', type=int, default=300,
                    help="Number of first video's frame to start with (default: 300)")
parser.add_argument('-check-frames', '--number-of-frames-to-check', type=int, default=600,
                    help="Number of frames to process (default: 600)")
parser.add_argument('--path-to-file', type=str, default='./metrics/FPS/',
                    help="Path to file to collect results")
parser.add_argument('-gpu','--gpu-id', type=int, default=0,
                    help="GPU's id to run the script on. Use -1 for CPU")
parser.add_argument('-b', '--batch-size', type=int, default=1,
                    help="Number of sequences the model processing at once (default: 1)")
parser.add_argument('-show', '--show-results', action=argparse.BooleanOptionalAction,
                    help="Use this argument to see the results after processing")
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction,
                    help="Use this argument to see the progress during processing frames")
parser.add_argument('-save', '--save-results', type=str, default="",
                    help="Path to save processed video. Don't use this parameter to not save the result (default: '')")
parser.add_argument("-dtype", "--tensor-dtype", default="float32", type=str,
                    help="Frames and model's data type")
