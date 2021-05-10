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

    # Initialize arguments based on dataset chosen
    if args.upan_data == "disjoint_mnist":
        args.arch = ["lenet5", "lenet5"]
    elif args.upan_data == "mnist_cifar10":
        args.arch = ["lenet5", "resnet18"]
    elif args.upan_data == "fmnist_kmnist":
        args.arch = ["resnet18", "resnet18"]

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
        m = fmnist_kmnist

    # Initialize logits statistics function
    experiment = m.smart_coordinator_upan

    # Pan settings
    if args.upan_type == "logits":
        upan_input_size = args.test_output_size
        upan_arch = PAN
    elif args.upan_type == "agnostic_logits":
        upan_input_size = 5
        upan_arch = AgnosticPAN

    # Running the test
    print(f"Testset: {args.testset}")
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
        upan = upan_arch(input_size=upan_input_size).to(args.device)
        upan.load_state_dict(
            torch.load(
                args.upan_dir
                + f"upan_{args.upan_type}_{args.upan_data}{args.arch}_{args.seeds[i]}",
                map_location=torch.device(args.device),
            )
        )
        result = experiment(args, all_experts[0], all_experts[1], upan, device, test_loaders)

        # Adding more info to the result to be saved
        for r in result:
            r.update({"iteration": i, "seed": args.seeds[i]})
        results.extend(result)

    # Save the results
    if args.save_results:
        save_results(
            f"upan_{args.upan_type}_{args.upan_data}_{args.testset}{args.test_arch}",
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
    parser.add_argument("--results_dir", type=str, default="./results/merge/smart_coord (upan)/")
    parser.add_argument("--model_dir", type=str, default="./cache/models/")
    parser.add_argument("--upan_dir", type=str, default="./cache/models/upan/")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.seeds = SEEDS
    args.device = device
    main(args)
