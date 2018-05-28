import argparse
import tensorflow as tf
from common.utils import check_data_folder

def parse_args():
    desc = "TensorFlow Implementation of GAN models"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='CGAN', help='The type of GAN', required=True)
    return check_args(parser.parse_args())

def check_args(args):
    check_data_folder()
    return args

def main():
    args = parse_args()
    print(args.gan_type)


if __name__ == "__main__":
    main()