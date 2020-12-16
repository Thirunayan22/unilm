# NOTE THE LAYOUT LM DATASET DOES NOT TAKE IN AN IMAGE IT TAKES IN TEXTS AND THE LOCATION OF THE TEXTS
# NOTE THE LAYOUT LM DATASET DOES NOT TAKE IN AN IMAGE IT TAKES IN TEXTS AND THE LOCATION OF THE TEXTS


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil
import torch
import inspect
import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from time import time
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from layoutlm import FunsdDataset, LayoutlmConfig, LayoutlmForTokenClassification
from run_model_sample import evaluate

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
}
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, LayoutlmConfig)
    ),
    (),
)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(data):
    batch = [i for i in zip(*data)]
    for i in range(len(batch)):
        if i < len(batch) - 2:
            batch[i] = torch.stack(batch[i], 0)
    return tuple(batch)


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


class CustomImageInference():
    def __init__(self, args, model, tokenizer, labels, pad_token_label_id):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.pad_token_label_id = pad_token_label_id
        self.mode = "test"

    def return_sample_output(self, run_infer=False,get_random=True):

        sample_dataset = FunsdDataset(
                self.args,self.tokenizer, self.labels, self.pad_token_label_id, mode=self.mode).__getitem__(np.random.randint(30))
            
        print("dataset size : ",len(next(iter(sample_dataset))))

        if(get_random):
            
            if(run_infer): #TODO  Remove  this run_infer condition and indent code below

                sample_item = next(iter(DataLoader([sample_dataset])))
            

                self.model.eval()
                with torch.no_grad():
                    inputs = {
                    "input_ids": sample_item[0].to("cuda"),
                    "attention_mask" : sample_item[1].to("cuda"),
                    "token_type_ids" : sample_item[2].to("cuda"),
                    "bbox" : sample_item[4].to("cuda")
                            }

                    print("INPUT ID SHAPE : ",inputs["input_ids"].shape)
                    print("ATTENTION MASK SHAPE : " ,inputs["attention_mask"].shape)
                    print("TOKEN SHAPE : ",inputs['token_type_ids'].shape)
                    print("BBOX SHAPE : " ,inputs['bbox'].shape)

                    print(inputs)
                    model_output = self.model(**inputs)
                return {"model_output": model_output}
        else:
            dataset_sampler = SequentialSampler(sample_dataset)
            sample_dataset = FunsdDataset(self.args,self.tokenizer, self.labels, self.pad_token_label_id, mode=self.mode)
            sample_dataloader = DataLoader(sample_dataset,sampler=dataset_sampler,collate_fn=None)

            for sample_item in sample_dataloader:
                    inputs = {
                    "input_ids": sample_item[0].to("cuda"),
                    "attention_mask" : sample_item[1].to("cuda"),
                    "token_type_ids" : sample_item[2].to("cuda"),
                    "bbox" : sample_item[4].to("cuda")
                            }

                    print("INPUT ID SHAPE : ",inputs["input_ids"].shape)
                    print("ATTENTION MASK SHAPE : " ,inputs["attention_mask"].shape)
                    print("TOKEN SHAPE : ",inputs['token_type_ids'].shape)
                    print("BBOX SHAPE : " ,inputs['bbox'].shape)

                    model_output = self.model(**inputs)
                    print({"model_output": model_output})
                    
                    print("model_output_shape:",model_output[0].shape)
                    logits = model_output[-1]
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = sample_item[3].to("cuda").detach().cpu().numpy()
            
            preds = np.argmax(preds,axis=2)
            label_map = {i:label for i,label in enumerate(self.labels)}
            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]
            print(out_label_list)
            print(preds_list)
            # print(out_label_ids)
            # print(label_map)
            print("=====================PREDICTIONS====================")
            print(preds.shape)
            print("=====================PREDICTIONS====================")
            print("label list: ",out_label_ids)
            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i,j] != self.pad_token_label_id:
                    
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
            
                        # print("EXCEPTION!!!!")
            pred_output= {"prediction_list":preds_list,"output_label_list":out_label_list}

            if(self.args.text_output):
                with open(f"{int(time())}_prediction_output.txt","w+",encoding="utf-8") as output_file:
                    output_file.write(str(pred_output))


            #NOTE: Convert model_output to readable output annotations from the model -  REFER : run_model_sample.py (lines 100-140)
            return pred_output
    
    def random_sample_item(self, run_infer=False,get_random=True):
        sample_item = FunsdDataset(self.args,self.tokenizer, self.labels, self.pad_token_label_id, mode=self.mode).__getitem__(np.random.randint(30))

        return sample_item

    def extract_features(self,image_path):
        #NOTE : Write code to extract necessary features from a given custom document image using pytesseract
        pass
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data_new/",
        type=str,
        required=False,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default="layoutlm",
        type=str,
        required=False,
        help="Model type selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="pretrained_model/",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="data_new/labels.txt",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )

    parser.add_argument(
        "--single_inference", type=str, default="", help="To pass through a single image"
    )

    parser.add_argument(
        "--text_output",type=bool, default=False, help="To output a prediction text file"
    )
    args = parser.parse_args()
    labels = get_labels(args.labels)
    num_labels = len(labels)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    print(args.local_rank)
    print(labels)
    # print(model.func_code.co_varnames)
    print("model params : ",inspect.signature(model))
    if args.local_rank in [-1,0]:
            pad_token_label_id = CrossEntropyLoss().ignore_index
            model.to("cuda")
            custom_image_inference = CustomImageInference(args,model,tokenizer,labels,pad_token_label_id)
            sample = custom_image_inference.return_sample_output(True,get_random=False)
            print(type(sample))
            print(sample)

    print("===========================================================================================")



if __name__ == '__main__':
    main()
