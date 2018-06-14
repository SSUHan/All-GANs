import argparse
import tensorflow as tf
from common.utils import check_data_folder, show_all_variables
from _tensorflow.CGAN import CGAN
from _tensorflow.ACGAN import ACGAN

def parse_args():
    desc = "TensorFlow Implementation of GAN models"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='CGAN', help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'celebA'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_point', type=int, default=300,
                        help='Directory name to save training logs')
    return check_args(parser.parse_args())


def check_args(args):
    check_data_folder()
    return args


def main():
    args = parse_args()
    print(args.gan_type)

    model_dict = {CGAN.model_name: CGAN,
                  ACGAN.model_name: ACGAN}
    gan = None
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        try:
            gan = model_dict[args.gan_type](sess,
                                            epoch=args.epoch,
                                            batch_size=args.batch_size,
                                            z_dim=args.z_dim,
                                            dataset_name=args.dataset,
                                            checkpoint_dir=args.checkpoint_dir,
                                            result_dir=args.result_dir,
                                            log_dir=args.log_dir,
                                            sample_point=args.sample_point)
        except KeyError as ke:
            print("[!] There is no option for {}".format(args.gan_type))
            exit(0)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training Finished!")


if __name__ == "__main__":
    main()
