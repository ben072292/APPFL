import time
import torch
import argparse
from mpi4py import MPI
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
import appfl.run_mpi as rm
import appfl.run_mpi_sync as rms
from losses.utils import get_loss
from models.utils import get_model
from metric.utils import get_metric
from dataloader.mnist_dataloader import get_mnist

"""
To run MPI with 5 clients:
mpiexec -np 6 python ./mnist_mpi_sync.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
"""

## read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")

## dataset
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--num_channel", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--num_pixel", type=int, default=28)
parser.add_argument("--model", type=str, default="CNN")
parser.add_argument("--partition", type=str, default="iid", choices=["iid", "class_noiid", "dirichlet_noiid"])
parser.add_argument("--seed", type=int, default=42)

## clients
parser.add_argument("--num_clients", type=int, default=-1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=3e-3)
parser.add_argument("--local_train_pattern", type=str, default="steps", choices=["steps", "epochs"], help="For local optimizer, what counter to use, number of steps or number of epochs")
parser.add_argument("--num_local_steps", type=int, default=100)
parser.add_argument("--num_local_epochs", type=int, default=1)

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--server_lr", type=float, default=0.01)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001)

## loss function
parser.add_argument("--loss_fn", type=str, required=False, help="path to the custom loss function definition file, use cross-entropy loss by default if no path is specified")
parser.add_argument("--loss_fn_name", type=str, required=False, help="class name for the custom loss in the loss function definition file, choose the first class by default if no name is specified")

## evaluation metric
parser.add_argument("--metric", type=str, default='metric/acc.py', help="path to the custom evaluation metric function definition file, use accuracy by default if no path is specified")
parser.add_argument("--metric_name", type=str, required=False, help="function name for the custom eval metric function in the metric function definition file, choose the first function by default if no name is specified")

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    assert comm_size > 1, "This script requires the toal number of processes to be greater than one!"
    args.num_clients = comm_size - 1 if args.num_clients <= 0 else args.num_clients

    ## Configuration
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.clientname = "ClientOptim" if args.local_train_pattern == "epochs" else "ClientStepOptim"
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs    

    ## server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs

    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = f"./outputs_{args.dataset}_{args.partition}_{args.num_clients}clients_{args.server}_{args.num_epochs}epochs_mpi"
    
    ## adaptive server
    cfg.fed.args.server_learning_rate = args.server_lr          # FedAdam
    cfg.fed.args.server_adapt_param = args.adapt_param          # FedAdam
    cfg.fed.args.server_momentum_param_1 = args.mparam_1        # FedAdam, FedAvgm
    cfg.fed.args.server_momentum_param_2 = args.mparam_2        # FedAdam

    start_time = time.time()

    ## User-defined model
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)

    ## User-defined data
    train_datasets, test_dataset = get_mnist(comm, cfg, partition=args.partition, visualization=True, seed=args.seed, alpha1=args.num_clients)

    ## Sanity check for the user-defined data
    if cfg.data_sanity == True:
        data_sanity_check(train_datasets, test_dataset, args.num_channel, args.num_pixel)

    print("-------Loading_Time=", time.time() - start_time)

    ## Running
    if args.num_clients == comm_size -1:
        if comm_rank == 0:
            rms.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
        else:
            rms.run_client(cfg, comm, model, loss_fn, train_datasets, test_dataset, metric)
    else:
        if comm_rank == 0:
            rm.run_server(cfg, comm, model, loss_fn, args.num_clients, test_dataset, args.dataset, metric)
        else:
            rm.run_client(cfg, comm, model, loss_fn, args.num_clients, train_datasets, test_dataset, metric)

    print("------DONE------", comm_rank)
        
if __name__ == "__main__":
    main()
