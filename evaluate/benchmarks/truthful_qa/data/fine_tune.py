"""
Run with ``python fine_tune.py --dataset finetune_truth.jsonl``
"""
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="finetune_truth.jsonl")
parser.add_argument("--file-id", type=str)
parser.add_argument("--model", type=str, default="babbage-002")
args = parser.parse_args()

client = OpenAI()
if args.file_id is None:
    file_id = client.files.create(
        file=open(args.dataset, "rb"),
        purpose="fine-tune"
    ).id
else:
    file_id = args.file_id

client.fine_tuning.jobs.create(
    training_file=file_id,
    model=args.model,
    hyperparameters={
        "n_epochs": 5,
        "batch_size": 21,
        "learning_rate_multiplier": 0.1
    }
)
client.models.list()