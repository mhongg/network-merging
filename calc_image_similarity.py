import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from config import Config
from utils import save_results
from skimage.metrics import structural_similarity as ssim
from mnist.dataloaders import mnist_first5_train_loader, mnist_last5_train_loader
from mnist_cifar10.dataloaders import mnist_train_loader, cifar10_single_channel_train_loader
from fmnist_kmnist.dataloaders import fmnist_train_loader, kmnist_train_loader

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB):
	# compute the mean squared error (the lower the error, the more "similar" the two images are)
    m = mse(imageA, imageB)
    # compute the structural similarity index (the closer to 1, the more "similar" the two images are)
    s = ssim(imageA, imageB, multichannel=True)
    return m, s


def calc_image_similarity(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True

    if args.dataset == "disjoint_mnist":
        dataloaders = [mnist_first5_train_loader(args.size), mnist_last5_train_loader(args.size)]
    elif args.dataset == "mnist_cifar10":
        dataloaders = [mnist_train_loader(args.size), cifar10_single_channel_train_loader(args.size)]
    elif args.dataset == "fmnist_kmnist":
        dataloaders = [fmnist_train_loader(args.size), kmnist_train_loader(args.size)]

    print(f"\nDataset: {args.dataset}")

    # Extract data from the first batch of both dataloaders
    for batches in zip(*dataloaders):
        images = []
        for data, _ in batches:
            images.append(data)
        break

    # Iterate and compare each pair of images
    total_mse, total_ssim = [], []
    for idx, (imageA, imageB) in enumerate(zip(*images)):
        imageA, imageB = imageA.permute(1, 2, 0), imageB.permute(1, 2, 0)
        m ,s = compare_images(imageA.numpy(), imageB.numpy())
        total_mse.append(m)
        total_ssim.append(s)
        
        if idx % args.log_interval == 0:
            print(f"Calculating image {idx}...")
    print(len(total_mse), len(total_ssim))
    avg_mse, avg_ssim = np.mean(total_mse), np.mean(total_ssim)
    print(f"Average MSE: {avg_mse} Average SSIM index: {avg_ssim}")

    # Save the results in list
    results = [{"mse": avg_mse, "ssim": avg_ssim,}]

    # Save all the results
    if args.save_results:
        save_results(
            f"image_similarity_{args.dataset}",
            results,
            args.results_dir,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_results", type=bool, default=Config.save_results)
    parser.add_argument("--results_dir", type=str, default="./results/experiments/image_similarity/")

    args = parser.parse_args()
    calc_image_similarity(args)