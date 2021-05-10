from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from utils import create_op_dir, save_results
from config import SEEDS, Config
from mnist.smart_coord import craft_disjoint_mnist_target
from mnist.dataloaders import (
    mnist_first5_train_loader_unseen_noshuffle, 
    mnist_last5_train_loader_unseen_noshuffle,
    mnist_first5_test_loader_unseen,
    mnist_last5_test_loader_unseen,
)
from mnist_cifar10.smart_coord import craft_mnist_cifar10_target
from mnist_cifar10.dataloaders import (
    mnist_single_channel_train_loader_unseen_noshuffle,
    mnist_3_channel_train_loader_unseen_noshuffle,
    mnist_single_channel_test_loader_unseen,
    cifar10_single_channel_train_loader_unseen_noshuffle,
    cifar10_3_channel_train_loader_unseen_noshuffle,
    cifar10_single_channel_test_loader_unseen,
)
from fmnist_kmnist.smart_coord import craft_fmnist_kmnist_target
from fmnist_kmnist.dataloaders import (
    fmnist_train_loader_unseen_noshuffle,
    kmnist_train_loader_unseen_noshuffle,
    fmnist_test_loader_unseen,
    kmnist_test_loader_unseen,
)
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN, compute_agnostic_stats
from train_upan import test as test_upan
from train_upan import eval_expert as upan_eval_expert
import time


def eval_expert(args, expert_idx, expert, device, data_loader):
    expert.eval()
    expert_output = []  # collect output logits of expert
    total_data = sum(len(data) for data, target in data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        logits = expert(data)    # output logits of expert
        expert_output.append(logits.detach())
        del logits

        if batch_idx % args.log_interval == 0:
            print(
                "Eval Expert: {} [{}/{} ({:.0f}%)]".format(
                    expert_idx + 1,
                    batch_idx * len(data),
                    total_data,
                    100.0 * batch_idx * len(data) / total_data,
                )
            )
    return expert_output


def eval_upan(args, upan, all_experts_output, device):
    upan.eval()
    fpan_target = []
    for expert_output in all_experts_output:
        upan_output = []
        for logits in expert_output:
            if args.upan_type == "logits":
                output = upan(logits)
            elif args.upan_type == "agnostic_logits":
                output = upan(compute_agnostic_stats(logits))
            output = F.log_softmax(output, dim=-1)
            upan_output.append(output)

        # Concatenate batches of UPAN outputs
        upan_output = torch.cat(upan_output)
        # Extract the output of UPAN (ie. probability of the expert truly belonging to the input data)
        upan_output = torch.index_select(upan_output, 1, torch.tensor([1]).to(device))
        upan_output = torch.flatten(upan_output)
        fpan_target.append(upan_output)

    # Concatenate UPAN predictions on different experts when given the same input data
    fpan_target = torch.stack(fpan_target, dim=1)
    # Extract index of the max log-probability (represents the expert chosen by UPAN)
    fpan_target = torch.argmax(fpan_target, dim=1)
    return fpan_target


def create_train_loader(args, device, all_experts, upan, train_loader):
    print("Obtaining FPAN training set:")
    # Using input data of the first dataloader as input data of FPAN
    fpan_input = []
    for data, target in train_loader[0]:
        fpan_input.append(data.to(device))
    fpan_input = torch.cat(fpan_input)

    # Feed data to each expert to collect their output logits
    all_experts_output = []
    for expert_idx, expert in enumerate(all_experts):
        expert_output = eval_expert(
            args,
            expert_idx,
            expert,
            device,
            train_loader[expert_idx],
        )
        all_experts_output.append(expert_output)

    # UPAN selects an expert as the target of FPAN
    fpan_target = eval_upan(args, upan, all_experts_output, device)

    # Slice dataset into the smallest unit
    fpan_train_data = [(fpan_input[idx], fpan_target[idx]) for idx in range(len(fpan_input))]

    return DataLoader(fpan_train_data, batch_size=args.batch_size, shuffle=True)


def create_test_loader(args, device, test_loader, task_idx):
    print("Obtaining FPAN test set: \n")
    fpan_input = []
    fpan_target = []
    # Using input data of the test loader as input data of FPAN, and generate target for testing
    for data, target in test_loader:
        fpan_input.append(data.to(device))
        fpan_target.append(torch.tensor([task_idx]*len(target)).to(device))
    fpan_input = torch.cat(fpan_input)
    fpan_target = torch.cat(fpan_target)

    # Slice dataset into smallest unit
    fpan_test_data = [(fpan_input[idx], fpan_target[idx]) for idx in range(len(fpan_input))]

    return DataLoader(fpan_test_data, batch_size=args.test_batch_size, shuffle=False)


def create_upan_test_loader(config_args, device, all_experts, test_loader, target_create_fn):
    # Feed test data to expert, return test set of upan
    print("Obtaining UPAN test set:")
    upan_test_loader = []
    for expert_idx, expert in enumerate(all_experts):
        # Directly append output of expert (not slicing into smallest unit)
        upan_test_loader.append(
            upan_eval_expert(
                config_args,
                expert_idx,
                expert,
                device,
                test_loader[expert_idx],
                target_create_fn[expert_idx]
            )
        )
    return upan_test_loader


def train(args, fpan, train_loaders, optimizer, epoch, task_idx):
    fpan.train()
    for batch_idx, batches in enumerate(zip(*train_loaders)):
        for data, target in batches:
            output = fpan(data)
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train FPAN with Task {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    task_idx + 1,
                    epoch,
                    batch_idx * len(data),
                    sum(len(i.dataset) for i in train_loaders),
                    100.0 * batch_idx * len(data) / sum(len(i.dataset) for i in train_loaders),
                    loss.item(),
                )
            )


def test(args, fpan, test_loader, task_idx):
    fpan.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = fpan(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print("Test set of task {}: Accuracy: {}/{} ({:.0f}%)".format(task_idx+1, correct, len(test_loader.dataset), acc))
    return test_loss, acc


def train_model(fpan, trial, device, expert_arch, train_loaders, test_loaders, target_create_fn, config_args):
    fpan = fpan.to(device)
    optimizer = optim.SGD(fpan.parameters(), lr=config_args.lr, momentum=config_args.momentum)

    # Load experts
    all_experts = []
    for expert_idx in range(len(expert_arch)):
        expert = expert_arch[expert_idx](
            input_channel=args.expert_input_channel[expert_idx], output_size=args.expert_output_size
        ).to(device)
        expert.load_state_dict(
            torch.load(
                args.model_dir +
                f"{args.expert[expert_idx]}_{args.expert_arch[expert_idx]}_{args.seeds[trial]}",
                map_location=torch.device(device),
            )
        )
        all_experts.append(expert)

    # Load UPAN model
    if args.upan_type == "logits":
        upan_input_size = args.model_output_size
        upan_arch = PAN
    elif args.upan_type == "agnostic_logits":
        upan_input_size = 5 # number of statistical functions used
        upan_arch = AgnosticPAN

    upan = upan_arch(input_size=upan_input_size).to(device)
    upan.load_state_dict(
        torch.load(
            args.upan_dir
            + f"upan_{args.upan_type}_{args.upan_data}{args.model_arch}_{args.seeds[trial]}",
            map_location=torch.device(device),
        )
    )
    upan_test_loader = create_upan_test_loader(config_args, device, all_experts, upan_test_loader, upan_target_create_fn)
    upan_test_loss, upan_acc = test_upan(config_args, device, upan, upan_test_loader)

    # Initialise dataloaders of FPAN
    fpan_train_loaders = []
    for train_loader in train_loaders:
        fpan_train_loaders.append(create_train_loader(config_args, device, all_experts, upan, train_loader))

    fpan_test_loaders =[]
    for task_idx, test_loader in enumerate(test_loaders):
        fpan_test_loaders.append(create_test_loader(config_args, device, test_loader, task_idx))

    for epoch in range(1, config_args.epochs + 1):
        start_time = time.time()
        # Train fpan
        train(config_args, fpan, fpan_train_loaders, optimizer, epoch, task_idx)
        # Test fpan
        all_loss = []
        all_acc = []
        for task_idx, fpan_test_loader in enumerate(fpan_test_loaders):
            test_loss, acc = test(config_args, fpan, fpan_test_loader, task_idx)
            all_loss.append(test_loss)
            all_acc.append(acc)
        avg_test_loss, avg_acc = np.mean(all_loss), np.mean(all_acc)
        print("Average loss = {:.6f} Average accuracy: {:.0f}%".format(avg_test_loss, avg_acc))
        print('Time taken: {}.\n'.format(time.time() - start_time))

    return fpan, avg_test_loss, avg_acc, upan_test_loss, upan_acc


def train_fpan(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on the chosen dataset
    if args.fpan_data == "disjoint_mnist":
        train_loaders = [
            [mnist_first5_train_loader_unseen_noshuffle(args.batch_size), mnist_first5_train_loader_unseen_noshuffle(args.batch_size)],
            [mnist_last5_train_loader_unseen_noshuffle(args.batch_size), mnist_last5_train_loader_unseen_noshuffle(args.batch_size)],
        ]
        test_loaders = [
            mnist_first5_test_loader_unseen(args.test_batch_size),
            mnist_last5_test_loader_unseen(args.test_batch_size),
        ]
        args.expert = ["first5_mnist", "last5_mnist"]
        args.expert_input_channel = [1, 1]
        args.expert_output_size = 5
        args.expert_arch = ["lenet5", "lenet5"]
        expert_arch = [LeNet5, LeNet5]
        target_create_fns = craft_disjoint_mnist_target
        fpan_output_size = 2
        fpan_input_channel = 1
    elif args.fpan_data == "mnist_cifar10":
        train_loaders = [
            [mnist_single_channel_train_loader_unseen_noshuffle(args.batch_size), mnist_3_channel_train_loader_unseen_noshuffle(args.batch_size)],
            [cifar10_single_channel_train_loader_unseen_noshuffle(args.batch_size), cifar10_3_channel_train_loader_unseen_noshuffle(args.batch_size)],
        ]
        test_loaders = [
            mnist_single_channel_test_loader_unseen(args.test_batch_size),
            cifar10_single_channel_test_loader_unseen(args.test_batch_size),
        ]
        args.expert = ["mnist", "cifar10"]
        args.expert_input_channel = [1, 3]
        args.expert_output_size = 10
        args.expert_arch = ["lenet5", "resnet18"]
        expert_arch = [LeNet5, ResNet18]
        target_create_fns = craft_mnist_cifar10_target
        fpan_output_size = 2
        fpan_input_channel = 1
    elif args.fpan_data == "fmnist_kmnist":
        train_loaders = [
            [fmnist_train_loader_unseen_noshuffle(args.batch_size), fmnist_train_loader_unseen_noshuffle(args.batch_size)],
            [kmnist_train_loader_unseen_noshuffle(args.batch_size), kmnist_train_loader_unseen_noshuffle(args.batch_size)],
        ]
        test_loaders = [
            fmnist_test_loader_unseen(args.test_batch_size),
            kmnist_test_loader_unseen(args.test_batch_size),
        ]
        args.expert = ["fmnist", "kmnist"]
        args.expert_input_channel = [1, 1]
        args.expert_output_size = 10
        args.expert_arch = ["resnet18", "resnet18"]
        expert_arch = [ResNet18, ResNet18]
        target_create_fns = craft_fmnist_kmnist_target
        fpan_output_size = 2
        fpan_input_channel = 1

    if args.fpan_arch == "resnet18":
        fpan_arch = ResNet18
    elif args.fpan_arch == "lenet5":
        fpan_arch = LeNet5
    elif args.fpan_arch == "lenet5_halfed":
        fpan_arch = LeNet5Halfed

    # Initialize arguments based on the dataset that UPAN was trained on
    if args.upan_data == "disjoint_mnist":
        args.model_arch = ["lenet5", "lenet5"]
        args.model_output_size = 5
    elif args.upan_data == "mnist_cifar10":
        args.model_arch = ["lenet5", "resnet18"]
        args.model_output_size = 10
    elif args.upan_data == "fmnist_kmnist":
        args.model_arch = ["resnet18", "resnet18"]
        args.model_output_size = 10

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"\nFPAN Dataset: {args.fpan_data}")
    print(f"FPAN arch: {args.fpan_arch}")
    print(f"UPAN Dataset: {args.upan_data}")
    print(f"UPAN type: {args.upan_type}\n")

    fpan_results = []
    for i in range(len(args.seeds)):
        print(f"Iteration {i+1}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])
        torch.cuda.manual_seed_all(args.seeds[i])
        torch.backends.cudnn.deterministic = True

        # Train FPAN model
        fpan, fpan_test_loss, fpan_acc, upan_test_loss, upan_acc = train_model(
            fpan=fpan_arch(input_channel=fpan_input_channel, output_size=fpan_output_size).to(device),
            trial=i,
            device=device,
            expert_arch=expert_arch,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            target_create_fn=target_create_fns,
            config_args=args,
        )

        # Save the FPAN model
        if args.save_models:
            torch.save(
                fpan.state_dict(),
                args.output_dir
                + f"fpan_{args.fpan_data}({fpan_input_channel})_({args.upan_data}_{args.upan_type}){args.seeds[i]}",
            )

        # Save the results in list first
        fpan_results.append(
            {
                "iteration": i,
                "seed": args.seeds[i],
                "loss": fpan_test_loss,
                "acc": fpan_acc,
                "upan loss": upan_test_loss,
                "upan acc": upan_acc,
            }
        )

    # Save all the results
    if args.save_results:
        save_results(
            f"fpan_{args.fpan_data}({fpan_input_channel})_({args.upan_data}_{args.upan_type})",
            fpan_results,
            args.results_dir,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--fpan_data",
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
        choices=["agnostic_logits", "feature", "logits"],
    )
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr, help="learning rate")
    parser.add_argument("--momentum", type=float, default=Config.momentum)
    parser.add_argument("--no_cuda", type=bool, default=Config.no_cuda)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--save_models", type=bool, default=False)
    parser.add_argument("--save_results", type=bool, default=Config.save_results)
    parser.add_argument("--results_dir", type=str, default="./results/experiments/fpan_unseen_unmix_batch_alternate/")
    parser.add_argument("--model_dir", type=str, default="./cache/models/")
    parser.add_argument("--upan_dir", type=str, default="./cache/models/upan/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/experiments/fpan_unseen_unmix_batch_alternate/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_fpan(args)
