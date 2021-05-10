from argparse import ArgumentParser
import numpy as np
import torch
from utils import save_results
import mnist
import mnist_cifar10
import fmnist_kmnist
from mnist.dataloaders import mnist_combined_test_loader
from mnist_cifar10.dataloaders import (
    mnist_cifar10_single_channel_test_loader,
    mnist_cifar10_3_channel_test_loader,
)
from fmnist_kmnist.dataloaders import fmnist_kmnist_test_loader
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN
from config import SEEDS, Config


def main(args):

    # Initialize arguments based on the chosen testset 
    if args.testset == "disjoint_mnist":
        test_loaders = [
            mnist_combined_test_loader(args.test_batch_size),
            mnist_combined_test_loader(args.test_batch_size),
        ]
        args.test_expert = ["first5_mnist", "last5_mnist"]
        args.test_input_channel = [1, 1]
        args.test_output_size = 5
        args.test_arch = ["lenet5", "lenet5"]
        test_arch = [LeNet5, LeNet5]
        fpan_input_channel = 1
        m = mnist
    elif args.testset == "mnist_cifar10":
        test_loaders = [
            mnist_cifar10_single_channel_test_loader(args.test_batch_size),
            mnist_cifar10_3_channel_test_loader(args.test_batch_size),
        ]
        args.test_expert = ["mnist", "cifar10"]
        args.test_input_channel = [1, 3]
        args.test_output_size = 10
        args.test_arch = ["lenet5", "resnet18"]
        test_arch = [LeNet5, ResNet18]
        fpan_input_channel = 1
        m = mnist_cifar10
    elif args.testset == "fmnist_kmnist":
        test_loaders = [
            fmnist_kmnist_test_loader(args.test_batch_size),
            fmnist_kmnist_test_loader(args.test_batch_size),
        ]
        args.test_expert = ["fmnist", "kmnist"]
        args.test_input_channel = [1, 1]
        args.test_output_size = 10
        args.test_arch = ["resnet18", "resnet18"]
        test_arch = [ResNet18, ResNet18]
        fpan_input_channel = 1
        m = fmnist_kmnist

    # Initialize logits statistics function
    experiment = m.smart_coordinator_fpan

    # FPAN architecture
    if args.fpan_arch == "resnet18":
        fpan_arch = ResNet18
    elif args.fpan_arch == "lenet5":
        fpan_arch = LeNet5
    elif args.fpan_arch == "lenet5_halfed":
        fpan_arch = LeNet5Halfed

    # Running the test
    print(f"Testset: {args.testset}")
    print(f"FPAN arch: {args.fpan_arch}")
    print(f"UPAN Dataset: {args.upan_data}")
    print(f"UPAN type: {args.upan_type}")
    results = []

    for i in range(len(args.seeds)):
        print(f"\nIteration: {i+1}, Seed: {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])
        torch.cuda.manual_seed_all(args.seeds[i])
        torch.backends.cudnn.deterministic = True

        # Load experts
        all_experts = []
        for expert_idx in range(len(test_arch)):
            expert = test_arch[expert_idx](
                input_channel=args.test_input_channel[expert_idx], output_size=args.test_output_size
            ).to(device)
            expert.load_state_dict(
                torch.load(
                    args.model_dir +
                    f"{args.test_expert[expert_idx]}_{args.test_arch[expert_idx]}_{args.seeds[i]}",
                    map_location=torch.device(device),
                )
            )
            all_experts.append(expert)

        # Running the experiment
        fpan = fpan_arch(input_channel=fpan_input_channel, output_size=len(args.test_expert)).to(args.device)
        fpan.load_state_dict(
            torch.load(
                args.fpan_dir
                + f"fpan_{args.testset}({fpan_input_channel})_({args.upan_data}_{args.upan_type}){args.seeds[i]}",
                map_location=torch.device(args.device),
            )
        )
        result = experiment(args, all_experts[0], all_experts[1], fpan, device, test_loaders)

        # Adding more info to the result to be saved
        for r in result:
            r.update({"iteration": i, "seed": args.seeds[i]})
        results.extend(result)

    # Save the results
    if args.save_results:
        save_results(
            f"fpan_{args.testset}({fpan_input_channel})_({args.upan_data}_{args.upan_type})",
            results,
            f"{args.results_dir}",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--testset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--fpan_arch",
        type=str,
        default="resnet18",
        choices=["lenet5", "lenet5_halfed", "resnet18"],
    )
    parser.add_argument(
        "--upan_data",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--upan_type",
        type=str,
        default="agnostic_logits",
        choices=["logits", "agnostic_logits"],
    )
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--no_cuda", type=bool, default=Config.no_cuda)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--results_dir", type=str, default="./results/merge/smart_coord (fpan)/")
    parser.add_argument("--model_dir", type=str, default="./cache/models/")
    parser.add_argument("--fpan_dir", type=str, default="./cache/models/fpan/")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.seeds = SEEDS
    args.device = device
    main(args)