import argparse


from engine import train_net


def get_parser():
    parser = argparse.ArgumentParser(
        description='Training phase'
    )

    parser.add_argument(
        "--image_scale",
        type=int,
        default=312,
        help='size of image after scaling (default: 312)'
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help='number of epochs to train (default: 30)'
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help='number of images used in an iteration (default: 8)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate (default: 1e-4)'
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default='ce',
        help='loss function (default: ce) [options: mse=mean square error, fl=focal loss]'
    )

    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default='weights',
        help='path to the directory for saving checkpoints (default: weights)'
    )

    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help='path to the checkpoint for loading (default: None)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    train_net(image_size=args.image_scale,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              lr=args.lr,
              criterion=args.criterion,
              save_checkpoint=args.save_checkpoint,
              load_checkpoint=args.load_checkpoint)
