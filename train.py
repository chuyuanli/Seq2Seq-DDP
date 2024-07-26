import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
import torch
import numpy as np
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import evaluate
import datasets
from datasets import load_dataset, concatenate_datasets

import pickle
import json
from nltk.tokenize import sent_tokenize
# nltk.download("punkt")
import time

from constant import *


def preprocess_function(samples, tokenizer, max_source_length, max_target_length, padding="max_length"):
    # add prefix to the input for t5
    input_str = ["discourse parsing: " + item for item in samples["dialogue"]]

    model_inputs = tokenizer(input_str, max_length=max_source_length, padding=padding, truncation=True, return_tensors="pt")

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(samples["structure"], max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Metric
try:
    # if error with loading rouge metric, download rouge.py and use it locally
    metric = datasets.load_metric(f"{ROOT_DIR}/rouge.py")
    print('load metric locally')
except:
    metric = evaluate.load("rouge")

# Helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    try:
        result = {k: round(v * 100, 4) for k, v in result.items()}
    except:
        result = {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def setup_tokenizer(cfg):
    local_model_path = os.path.join(HF_MODEL_DIR, "models--" + "--".join(cfg.pretrained_model_name.split("/")), "snapshots/model")
    print(f"Read hf tokenizer from {local_model_path}")
    
    if cfg.t5_family in ['flan-t5', 't5']:
        tokenizer = T5Tokenizer.from_pretrained(local_model_path, local_files_only=True)
    elif cfg.t5_family == 't0':
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    
    # update tokenizer with special tokens
    if cfg.structure_type == "natural":
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
    elif cfg.structure_type == "labelmasked":
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
        special_tokens += [f"rel{i}" for i in range(16)] #masked 16 relation labels
    elif cfg.structure_type == "augmented":
        special_tokens = ["[", "]", "|", "="]
        special_tokens += [f"edu{i}" for i in range(MAX_EDU_LEN)]
    elif cfg.structure_type in ["focus"]: 
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
        special_tokens += ["|", "**"]
    elif cfg.structure_type in ["natural2"]: #transition-based natural
        special_tokens = [f"[edu{i}]" for i in range(MAX_EDU_LEN)]
        special_tokens += ["[", "]"]
    tokenizer.add_tokens(special_tokens)
    
def train(model, tokenizer, train_data, dev_data, out_dir, cfg):
    """Set up trainer"""
    
    repository_id = f"{ROOT_DIR}/{cfg.pretrained_model_name.split('/')[1]}-stac-train"
    
    # TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        learning_rate=float(cfg.lr),
        per_device_train_batch_size=cfg.batchsize,
        per_device_eval_batch_size=cfg.batchsize,
        gradient_accumulation_steps=1, # optimize vram
        gradient_checkpointing=True,
        optim="adamw_torch", # "adamw_torch" | "adafactor", "adamw_bnb_8bit" 
        fp16=False, # default False, whether use fp16 16-bit (mixed) precision training instead of 32-bit training.
        bf16=True if cfg.bfloat16 else False, #default False, Requires Ampere or higher NVIDIA architecture or using CPU (use_cpu) or Ascend NPU.
        predict_with_generate=True,
        num_train_epochs=cfg.epoch,
        evaluation_strategy="epoch",
        eval_steps=cfg.step,
        logging_dir=f"{repository_id}/logs",
        logging_strategy='steps',
        logging_steps=cfg.step,
        save_strategy="epoch",
        save_steps=cfg.step,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
                    tokenizer,
                    model=model,
                    label_pad_token_id=label_pad_token_id,
                    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data, #train_data['train']
        eval_dataset=dev_data, #train_data['test']
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    return trainer

def exe_train(trainf, devf, tokenizer, cfg):
    """Execute training / fine-tuning.

    Args:
        trainf (str): train json file path
        devf (str): dev json file path
        cfg (str): arguments
    """
    t = time.time()
    
    base_train = load_dataset('json', data_files=trainf)['train'] 
    data_dev = load_dataset('json', data_files=devf)['train']
    print(len(base_train['dialogue']), len(data_dev['dialogue']))

    local_model_path = os.path.join(HF_MODEL_DIR, "models--" + "--".join(cfg.pretrained_model_name.split("/")), "snapshots/model")
    print(f"read huggingface model from {local_model_path}")
    
    tokenized_inputs = concatenate_datasets([base_train, data_dev]).map(
                            lambda x: tokenizer(x["dialogue"], truncation=False), 
                            batched=True, 
                            remove_columns=["dialogue", "structure"],
                            )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Train {cfg.train_corpus} {cfg.structure_type} format max input length: {max_source_length}")

    tokenized_targets = concatenate_datasets([base_train, data_dev]).map(
                            lambda x: tokenizer(x["structure"], truncation=False), 
                            batched=True, 
                            remove_columns=["dialogue", "structure"]
                            )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Train {cfg.train_corpus} {cfg.structure_type} format max output length: {max_target_length}")
    
    # tokenize train and dev
    tokenized_train = base_train.map(preprocess_function, 
                                    fn_kwargs={"tokenizer": tokenizer},
                                    batched=True, 
                                    remove_columns=["dialogue", "structure", "id"])
    tokenized_dev = data_dev.map(preprocess_function, 
                                    fn_kwargs={"tokenizer": tokenizer},
                                    batched=True,
                                    remove_columns=["dialogue", "structure", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_train.features)}")                        

    # set up model
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path,
                                        local_files_only=True,
                                        torch_dtype=torch.bfloat16 if cfg.bfloat16 else torch.float32, #torch.float16 or torch.bfloat16 or torch.float, default load torch.float (fp32)
                                        device_map="auto" # pip install accelerate. torchrun .py
                                        )
    model.resize_token_embeddings(len(tokenizer))
    
    # path to store fine-tuned model
    model_dir = os.path.join(FT_MODEL_DIR, f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_{cfg.structure_type}_seed{cfg.seed}_{cfg.lr}")
    
    # start train
    trainer = train(model, tokenizer, tokenized_train, tokenized_dev, out_dir=model_dir, cfg=cfg)

    # record train set and ft result
    train_dev = {'trainset': base_train, 'devset': data_dev, 'losslog': trainer.state.log_history}
    pif = os.path.join(model_dir, 'traininglog')
    with open(pif, 'wb') as outf:
        pickle.dump(train_dev, outf)
        
    print(f"time {time.time()-t}, train time/doc : {(time.time()-t)/len(base_train['dialogue'])}")
                    
def exe_test(testf, device, cfg):
    """Execute prediction.

    Args:
        testf (str): test json file path
        device (str): GPU or CPU
        cfg (str): arguments
    """
    t = time.time()
    
    # load test dataset
    data_test = load_dataset('json', data_files=testf)['train']
    print(len(data_test['dialogue'])) 

    # load tokenizer
    model_dir = os.path.join(FT_MODEL_DIR, f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_{cfg.structure_type}_seed{cfg.seed}_{cfg.lr}")
    fn_model_name = f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_{cfg.structure_type}_seed{cfg.seed}_{cfg.lr}"

    modelcheckpoint = os.path.join(model_dir, MODEL2CHECKPOINT[fn_model_name])
    tokenizer = AutoTokenizer.from_pretrained(modelcheckpoint, local_files_only=True)                   
    model = AutoModelForSeq2SeqLM.from_pretrained(modelcheckpoint, local_files_only=True,\
                                                torch_dtype=torch.bfloat16 if cfg.bfloat16 else torch.float32)
    
    model.parallelize()
    
    # load string for inference
    input_str = ["discourse parsing: " + item for item in data_test["dialogue"]]
    
    tokenized_test = tokenizer(input_str,
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt"
                                ).input_ids.to(device)
    
    max_input_length = max([len(x) for x in tokenized_test])
    print(f"Test {structure_type} format max input length: {max_input_length}")
    
    decoded_preds = []
    if cfg.structure_type == 'augmented':
        max_infer_len = 1024
    else: 
        max_infer_len = 512
        
    # predict_results = model.generate(tokenized_test, max_new_tokens=max_infer_len) # if VRAM big enough, batch decode
    # decoded_preds = tokenizer.batch_decode(predict_results, skip_special_tokens=True)
    for txt in tokenized_test: #if VRAM OOM, predict example one by one
        predict_result = model.generate(input_ids=txt.unsqueeze(0), max_new_tokens=max_infer_len)
        decoded_preds.append(tokenizer.decode(predict_result[0], skip_special_tokens=True))   
                          
    # log prediction
    outfile_name = f"{cfg.t5_family}-{cfg.model_size}_train_{cfg.train_corpus}_test_{cfg.test_corpus}_{cfg.structure_type}_seed{cfg.seed}_gen{max_infer_len}_lr{cfg.lr}.jsonl"

    res_file = os.path.join(ROOT_DIR, f"generation/{outfile_name}")
    
    with open(res_file, 'w') as of:
        for i, s in enumerate(decoded_preds):
            result = {'id': data_test["id"][i], "gen_output": s}
            json.dump(result, of)
            of.write('\n')
            
    print(f"time {time.time()-t}, infer time/doc : {(time.time()-t)/len(data_test['dialogue'])}")
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()            
    
    parser.add_argument("--train_corpus", type=str, default="stac", help="train corpus: stac, molweni")
    parser.add_argument("--test_corpus", type=str, default="stac", help="test corpus: stac, molweni")
    parser.add_argument("--do_train", action="store_true", default=False, help="if do train")
    parser.add_argument("--do_test", action="store_true", default=False, help="if do test")
    parser.add_argument("-s", "--structure_type", type=str, default=None, required=True, \
                        help="end2end: 'natural', 'augmented', 'labelmasked' | transition-based: 'focus', 'natural2'.")
    parser.add_argument("-t", "--t5_family", type=str, default="t0-3b", help="choose from: 't0-3b', 'flan-t5', 't5'")  
    parser.add_argument("-m", "--model_size", type=str, default="3b", \
                        help="choose from: flan-t5: 'base', 'large', 'xl' 3B, 'xxl' 11B | t0: 3b, 11b, pp | t5: 3b, large")  
    parser.add_argument("-b", "--bfloat16", action="store_true", default=False, help="if do bfloat16, default False")  
    parser.add_argument("-l", "--lr", type=str, default='2e-5', help="5e-5 up to xl/3b | 2e-5 xxl/11b")  
    parser.add_argument("-e", "--epoch", type=int, default=5, help="3b models: stac 10 epoch, molweni 3 epoch")  
    parser.add_argument("--batchsize", type=int, default=4, help="t0-3b: 4, flan-t5-base and large: 8")  
    parser.add_argument("--step", type=int, default=2000, help="2000 for molweni transition-based (focus, natural2) | 500 for all else")  
    parser.add_argument("--seed", type=int, default=27, help="seed: 27, 16, etc")
    args = parser.parse_args()
    
    train_corpus = args.train_corpus 
    test_corpus = args.test_corpus
    structure_type = args.structure_type
    
    MAX_EDU_LEN = 37 # stac: 37, molweni: 14
                        
    # choose a model from t5 family
    t5_family = args.t5_family
    assert t5_family in ['t0-3b', 'flan-t5', 't5'], "Choose from {'t0-3b', 'flan-t5', 't5'}."
    model_size = args.model_size
    namematch = {"t0-3b": f"bigscience/T0_3B",
                "flan-t5": f"google/flan-t5-{model_size}",
                "t5": f"google-t5/t5-{model_size}"}
    pretrained_model_name = namematch[t5_family]
        
    # load train, dev, test
    trainf = f"{ROOT_DIR}/data/{train_corpus}_{structure_type}_train.json"
    devf = f"{ROOT_DIR}/data/{train_corpus}_{structure_type}_dev.json" 
    testf = f"{ROOT_DIR}/data/{test_corpus}_{structure_type}_test.json" 

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    set_seed(seed=args.seed)
    
    # set up tokenizer
    tokenizer = setup_tokenizer(cfg=args)
    
    if args.do_train:  
        exe_train(trainf, devf, tokenizer, cfg=args)

    if args.do_test:  
        exe_test(testf, device, cfg=args)
    