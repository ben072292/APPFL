import time
import torch
import argparse
import appfl.run_mpi as rm
import appfl.run_mpi_sync as rms
from mpi4py import MPI
from appfl.config import *
from appfl.misc.data import *
from appfl.misc.utils import *
from models.utils import get_model
from losses.utils import get_loss
from metric.utils import get_metric
from dataloader.cifar10_dataloader import get_cifar10

#additional import for vision transformer
import torchvision
import torchvision.transforms as transforms

#profiling imports
import cProfile
import pstats

"""
To run MPI with 2 clients:
mpiexec -np 3 python ./cifar10_mpi_sync_Vit.py --partition iid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10 --train_data_batch_size 8
"""
 
## read arguments  
parser = argparse.ArgumentParser() 
parser.add_argument('--device', type=str, default="cpu")    

## dataset and model
parser.add_argument('--dataset', type=str, default="CIFAR10")   
parser.add_argument('--num_channel', type=int, default=3)   
parser.add_argument('--num_classes', type=int, default=10)   
parser.add_argument('--num_pixel', type=int, default=32)   
# parser.add_argument('--model', type=str, default="resnet18")   
parser.add_argument('--model', type=str, default="vit")  
parser.add_argument('--train_data_batch_size', type=int, default=128)   
parser.add_argument('--test_data_batch_size', type=int, default=128)  
parser.add_argument("--partition", type=str, default="iid", choices=["iid", "class_noiid", "dirichlet_noiid"])
parser.add_argument("--seed", type=int, default=42) 

## clients
parser.add_argument("--num_clients", type=int, default=-1)
parser.add_argument('--client_optimizer', type=str, default="Adam")    
parser.add_argument('--client_lr', type=float, default=1e-3)    
parser.add_argument("--local_train_pattern", type=str, default="steps", choices=["steps", "epochs"], help="For local optimizer, what counter to use, number of steps or number of epochs")
parser.add_argument("--num_local_steps", type=int, default=10)
parser.add_argument("--num_local_epochs", type=int, default=1)
parser.add_argument("--do_validation", action="store_true")

## server
parser.add_argument('--server', type=str, default="ServerFedAvg")    
parser.add_argument('--num_epochs', type=int, default=1)    
parser.add_argument("--server_lr", type=float, default=0.1)
parser.add_argument("--mparam_1", type=float, default=0.9)
parser.add_argument("--mparam_2", type=float, default=0.99)
parser.add_argument("--adapt_param", type=float, default=0.001) 

## privacy preserving
parser.add_argument("--use_dp", action="store_true", default=False, help="Whether to enable differential privacy technique to preserve privacy")
parser.add_argument("--epsilon", type=float, default=1, help="Privacy budget - stronger privacy as epsilon decreases")
parser.add_argument("--clip_grad", action="store_true", default=False, help="Whether to clip the gradients")
parser.add_argument("--clip_value", type=float, default=1.0, help="Max norm of the gradients")
parser.add_argument("--clip_norm", type=float, default=1, help="Type of the used p-norm for gradient clipping")
 
## loss function
parser.add_argument("--loss_fn", type=str, required=False, help="path to the custom loss function definition file, use cross-entropy loss by default if no path is specified")
parser.add_argument("--loss_fn_name", type=str, required=False, help="class name for the custom loss in the loss function definition file, choose the first class by default if no name is specified")

## evaluation metric
parser.add_argument("--metric", type=str, default='metric/acc.py', help="path to the custom evaluation metric function definition file, use accuracy by default if no path is specified")
parser.add_argument("--metric_name", type=str, required=False, help="function name for the custom eval metric function in the metric function definition file, choose the first function by default if no name is specified")

# additional arguments for vit
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
# parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32", type=int)
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()    

if torch.cuda.is_available():
    args.device="cuda"

## Run
def main():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size() 
    args.num_clients = comm_size - 1 if args.num_clients <= 0 else args.num_clients    

    ## Configuration     
    cfg = OmegaConf.structured(Config) 
    cfg.device = args.device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

    ## dataset
    cfg.train_data_batch_size = args.train_data_batch_size
    cfg.test_data_batch_size = args.test_data_batch_size
    cfg.train_data_shuffle = True

    ## clients
    cfg.num_clients = args.num_clients
    cfg.fed.clientname = "ClientOptim" if args.local_train_pattern == "epochs" else "ClientStepOptim"
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_steps = args.num_local_steps
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.validation = args.do_validation
    
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

    ## privacy preserving
    cfg.fed.args.use_dp = args.use_dp
    cfg.fed.args.epsilon = args.epsilon
    cfg.fed.args.clip_grad = args.clip_grad
    cfg.fed.args.clip_value = args.clip_value
    cfg.fed.args.clip_norm = args.clip_norm
    
    start_time = time.time()

    ## User-defined model
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)

    ## User-defined data
    train_datasets, test_dataset = get_cifar10(
        comm, 
        num_clients=cfg.num_clients,
        partition=args.partition, 
        visualization=True, 
        output_dirname=cfg.output_dirname, 
        seed=args.seed, 
        alpha1=args.num_clients
    )

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

# if __name__ == "__main__":
#     main()

# Run profiling on all nodes
# if __name__ == "__main__":
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     profile_filename = f"cifar10_mpi_sync_profile_ViT_{rank}.prof"
    
#     # Run profiling
#     with cProfile.Profile() as pr:
#         main()
#     pr.dump_stats(profile_filename)

#     # Ensure the profile file is properly closed
#     if os.path.exists(profile_filename):
#         # Analyze the profiling results
#         p = pstats.Stats(profile_filename)
#         p.sort_stats('cumulative').print_stats(10)
#     else:
#         print(f"Profile file for rank {rank} was not created.")

# Only run profiling in rank 0 (server) and rank 1 (client 1)
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Run profiling only for ranks 0 and 1
    if rank in [0, 1]:
        profile_filename = f"cifar10_mpi_sync_profile_ViT_{rank}_{size}.prof"
        
        # Run profiling
        with cProfile.Profile() as pr:
            main()
        pr.dump_stats(profile_filename)

        # Ensure the profile file is properly closed and analyze the results
        if os.path.exists(profile_filename):
            # Analyze the profiling results
            p = pstats.Stats(profile_filename)
            p.sort_stats('cumulative').print_stats(10)
        else:
            print(f"Profile file for rank {rank} was not created.")
    else:
        # For other ranks, simply run the main function without profiling
        main()
