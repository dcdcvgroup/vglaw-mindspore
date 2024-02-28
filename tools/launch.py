import os
import sys
import time
import signal
import subprocess
from argparse import ArgumentParser, REMAINDER

parser = ArgumentParser(description="PyTorch distributed training launch helper utility that will spawn up multiple distributed processes")

parser.add_argument("--nnodes", type=int, default=1, help="The number of nodes to use for distributed training")
parser.add_argument("--node_rank", type=int, default=0, help="The rank of the node for multi-node distributed training")
parser.add_argument("--nproc_per_node", type=int, default=-1, help="The number of processes to launch on each node for GPU training, this is recommended "
                                                                   "to be set to the number of GPUs in your system so that each process can be bound to a single GPU.")
parser.add_argument("--master_addr", default="127.0.0.1", type=str, help="Master node (rank 0)'s address, should be either the IP address or the hostname of node 0, "
                                                                         "for single node multi-proc training, the --master_addr can simply be 127.0.0.1")
parser.add_argument("--master_port", default=29500, type=int, help="Master node (rank 0)'s free port that needs to be used for communication during distributed training")
parser.add_argument("-m", "--module", default=False, action="store_true", help="Changes each process to interpret the launch script as a python module, executing with the " 
                                                                               "same behavior as 'python -m'.")
parser.add_argument("--executable", default=None, type=str, help="python executable path")
parser.add_argument("training_script", type=str, help="The full path to the single GPU training program/script to be launched in parallel, "
                                                      "followed by all the arguments for the training script")
parser.add_argument('training_script_args', nargs=REMAINDER)  # rest from the training program

args = parser.parse_args()

args.master_addr = os.environ.get('MASTER_IP', args.master_addr)
args.master_port = os.environ.get('MASTER_PORT', args.master_port)
args.nnodes = int(os.environ.get('WORLD_SIZE', args.nnodes))
args.node_rank = int(os.environ.get('RANK', args.node_rank))
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    args.nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
elif args.nproc_per_node < 0:
    args.nproc_per_node = len([d for d in subprocess.check_output("nvidia-smi -L", shell=True).decode().split('\n') 
                               if d.startswith('GPU')])

if args.executable is None:
    args.executable = sys.executable
# world size in terms of number of processes
dist_world_size = args.nproc_per_node * args.nnodes
# set PyTorch distributed related environmental variables
current_env = os.environ.copy()
current_env["MASTER_ADDR"] = args.master_addr
current_env["MASTER_PORT"] = str(args.master_port)
current_env["WORLD_SIZE"] = str(dist_world_size)
current_env["OMP_NUM_THREADS"] = str(1)

# create, open and add subprocess
processes = []
for local_rank in range(args.nproc_per_node):
    # each process's rank
    dist_rank = args.nproc_per_node * args.node_rank + local_rank
    current_env["RANK"] = str(dist_rank)
    current_env["LOCAL_RANK"] = str(local_rank)
    # spawn the processes
    cmd = [args.executable, "-u"]
    if args.module:
        cmd.append("-m")
    cmd.append(args.training_script)
    cmd.append("--local_rank={}".format(local_rank))
    cmd.extend(args.training_script_args)

    process = subprocess.Popen(cmd, env=current_env)
    processes.append(process)

# create sigkill handler 
sig_names = {2: "SIGINT", 15: "SIGTERM"}
last_return_info = None  # tuple: (local_rank, return_code)

def sigkill_handler(signum, frame):
    for process in processes:
        print(f"Killing subprocess {process.pid}")
        try:
            process.kill()
        except Exception:
            pass
    if last_return_info is not None:
        print(f'Process(rank[{last_return_info[0]}]) exited unexpectedly!')
        raise subprocess.CalledProcessError(returncode=last_return_info[1], cmd=cmd)
    if signum in sig_names:
        print(f"Main process received {sig_names[signum]}, exiting")
    sys.exit(1)

# pass SIGINT/SIGTERM to children if the parent is being terminated
signal.signal(signal.SIGINT, sigkill_handler)
signal.signal(signal.SIGTERM, sigkill_handler)

# poll all the subprocesses until normal/abnormal exit
process_to_rank = {proc: rank for rank, proc in enumerate(processes)}
alive_processes = set(processes)
while len(alive_processes):
    finished_processes = []
    for process in alive_processes:
        if process.poll() is None:
            # the process is still running
            continue
        else:
            if process.returncode != 0:
                last_return_info = (process_to_rank[process], process.returncode)  # for sigkill_handler
                sigkill_handler(signal.SIGTERM, None)  # not coming back
            else:
                finished_processes.append(process)  # exited cleanly
    alive_processes = set(alive_processes) - set(finished_processes)
    time.sleep(1)

