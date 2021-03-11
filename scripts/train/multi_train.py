import json, os, traceback
from argparse import ArgumentParser
import subprocess

parser = ArgumentParser(description='Multiple language trainer')
parser.add_argument('--task', action='store', dest='task',
                    help='Type of task : "tokenizer", "lemmatizer", "cwe", "tagger", "parser"')
parser.add_argument('--train', action='store', dest='train_file',
                    help='Start building a tagger model')
parser.add_argument('--patience', action='store', type=int, default=20, dest='patience',
                    help='Number of epochs before early stopping (default=20)')
parser.add_argument('--store_folder', action='store', dest='store_folder', help='Output folder')
parser.add_argument('--gpus', action='store', dest='gpus', type=str,
                    help='How many GPUs to use (default=1)', default="1")
parser.add_argument('--num-workers', action='store', dest='num_workers', type=int,
                    help='How many dataloaders to use (default=4)', default=4)
parser.add_argument('--accelerator', action='store', type=str, default="ddp", dest='accelerator',
                    help='Accelerator (see Pytorch Lightning accelerators)')
parser.add_argument('--batch-size', action='store', type=int, default=16, dest='batch_size',
                    help='Batch size (default=16)')
parser.add_argument('--grad-acc', action='store', type=int, default=1, dest='grad_acc',
                    help='Gradient accumulation steps (default=1)')
parser.add_argument('--debug', action='store_true', dest='debug',
                    help='Do some standard stuff to debug the model')
parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training')
parser.add_argument('--lm-model', action='store', dest='lm_model',
                    help='What LM model to use (default=xlm-roberta-base)')
parser.add_argument('--lm-device', action='store', dest='lm_device', default='cuda:0',
                    help='Where to load LM (default=cuda:0)')
parser.add_argument('--config', action='store', dest='config_file', help='Load config file')

parser.add_argument('--force_all', action='store_true', dest='force_all', help='Overwrite everything')
parser.add_argument('--retry_failed', action='store_true', dest='retry_failed', help='Retry unfinished training runs')
parser.add_argument('--yaml_folder', action='store', dest='yaml_folder', help='Where the yaml config files are stored')
parser.add_argument('--suffix', action='store', dest='suffix', help='Model name suffix')
args = parser.parse_args()


print(f"Running {args.task} on {args.yaml_folder}")

# load training status
if os.path.exists(f"{args.task}-status.json"):
    with open(f"{args.task}-status.json", "r") as f:
        jobs = json.load(f)
else:
    jobs = {}

# list all yaml files in folder
files = os.listdir(args.yaml_folder)
yamls = {}
for file in files:
    if ".yaml" in file:
        yamls[file.replace(".yaml","")] = os.path.join(args.yaml_folder, file)

for yaml in yamls:
    do_it = False
    # skip when not force_all, or when it is failed and rety_failed is True
    if yaml in jobs and jobs[yaml]!="Done" and args.retry_failed is True:
        do_it = True
    if args.force_all == True:
        do_it = True
    if yaml not in jobs:
        do_it = True
    if not do_it:
        print(f"Skipping job {yaml} ...\n")
        continue

    store = os.path.join(args.store_folder, f"{yaml}-{args.suffix}-{args.task}")
    print("\n\n")
    print("_"*80)
    print(f"Running job {yaml} with model store = {store}")

    jobs[yaml]="Training"
    with open(f"{args.task}-status.json", "w") as f:
        json.dump(jobs, f)
    try:
        subprocess.run(["python3", "cube3/trainer.py",
                        "--task", f"{args.task}",
                        "--train", f"{yamls[yaml]}",
                        "--store", f"{store}",
                        "--gpus", f"{args.gpus}",
                        "--num-workers", f"{args.num_workers}",
                        "--accelerator", f"{args.accelerator}",
                        "--batch-size", f"{args.batch_size}",
                        "--grad-acc", f"{args.grad_acc}",
                        #"--resume", f"{args.resume}",
                        "--lm-model", f"{args.lm_model}",
                        "--lm-device", f"{args.lm_device}"], check=True)

        jobs[yaml] = "Done"
        with open(f"{args.task}-status.json", "w") as f:
            json.dump(jobs, f)

    except Exception as ex:
        print(f"Job has failed: {ex}")
        #traceback.print_stack()
        import sys
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info








