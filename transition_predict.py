import os
import time
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import set_seed
from collections import defaultdict

from constant import *


class State(object):
    """document parsing state"""
    
    def __init__(self, input_document, structure_type, model_dir, fn_model_name, 
                 slide_window=True, max_len_doc=18, fix_count=False, bfloat16=True) -> None:
        """Create state object to process document.
        """
        self.structure_type = structure_type
        self.slide_window = slide_window
        self.fix_count = fix_count
        self.max_len_doc = max_len_doc
        self.done = False
        self.prefix = "dicourse parsing: "
        
        self.edu_map, self.edu_map_context = -1, [] # edu index
        self.edu, self.edu_context = "", [] # edu text
        self.annotation, self.annotation_context = "", []
        self.input_annotation, self.input_annotation_context = "", []
        self.prediction_str = {} # keep the predictions for each edu, {0: '[edu0] is root'...}, return in the end
        self.miss_count = defaultdict(list) #record miss head count in prediction
        self.fail_parse = 0
        
        self._read_input_doc(input_document)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bfloat16 = bfloat16
        self._load_trained_model(model_dir, fn_model_name)
        
    def _read_input_doc(self, doc_dict):
        """read input document object, fill in edu_map_context and edu_context"""
        self.docid = doc_dict['id']
        self.edu_map_context = doc_dict['edu_maps']
        self.edu_context = doc_dict['edus']
        self.max_edu_map = len(self.edu_map_context) #longest edu in the doc
    
    def _load_trained_model(self, model_dir, fn_model_name):
        self.modelcheckpoint = os.path.join(model_dir, MODEL2CHECKPOINT[fn_model_name])
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelcheckpoint, local_files_only=True)                   
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.modelcheckpoint, local_files_only=True,\
                                                    torch_dtype=torch.bfloat16 if self.bfloat16 else torch.float32,
                                                    # device_map="auto"
                                                    )
        self.model.parallelize()
        # self.model.to(self.device)
    
    def get_focus_input_annotation(self):
        """prepare input string for prediction"""
        if self.annotation == "": #first step, no previous annotation stored
            newinput = f"** [edu{self.edu_map}] {self.edu}"
        else:
            if self.slide_window and self.edu_map > self.max_len_doc-1:
                prev_context = self.annotation_context[-1].split(' ; ')
                slided_context = ' ; '.join(prev_context[1:]).strip()
                newinput = slided_context.replace("**", "") + ' ; ' + f"** [edu{self.edu_map}] {self.edu}"
            else:
                newinput = self.annotation_context[-1].replace("**", "").strip() + ' ; ' + f"** [edu{self.edu_map}] {self.edu}"
        return newinput

    def get_natural2_input_annotation(self):
        """prepare input string for prediction"""
        if self.annotation == "": #first step, no previous annotation stored
            newinput = f"[edu{self.edu_map}] [{self.edu}] is"
        else:
            if self.slide_window and self.edu_map > self.max_len_doc-1:
                prev_context = self.annotation_context[-1].split('; ')
                slided_context = ' ; '.join(prev_context[1:]).strip()
                newinput = slided_context + ' ; ' + f"[edu{self.edu_map}] [{self.edu}] is"
            else:
                newinput = self.annotation_context[-1].strip() + ' ; ' + f"[edu{self.edu_map}] [{self.edu}] is"
        return newinput
    
    def encode(self, annotation_str):
        """encode an input string"""
        return self.tokenizer(annotation_str, #only need to tokenize string, no need to call preprocess function
                            # max_length=max_source_length, 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt"
                            ).input_ids.to(self.device)
    
    def predict(self, encoded_str):
        """use loaded model and encoded string to predict a sequence, which is the raw y"""
        predict_result = self.model.generate(input_ids=encoded_str, max_new_tokens=512)
        raw_y = self.tokenizer.decode(predict_result[0], skip_special_tokens=True)
        return raw_y #eg: [edu7] is Acknowledgement of [edu6] Acknowledgement of [edu5] ; EOD
    
    def _postprocess_focus_y_for_input_annotation(self, y):
        """post process prediction string y in order to put in next step input annotation"""
        if y.startswith("root"):
            returny = 'root'
        else:
            elements = y.replace('of', ' ').split()
            returny = ""
            if len(elements) % 2 == 0:
                for j in range(0, len(elements), 2):
                    returny += f"{elements[j]} of {elements[j+1]} "
            else:
                # failed case, e.g.: elements = ['Conditional', 'Continuation', '[edu1]']
                returny = f"{elements[0]} of {elements[-1]}" #take the first and the last ele in the list
                self.fail_parse += 1
                # raise ValueError(f"Doc {self.docid} incomplete prediction: {elements}")
        return returny.strip()   
    
    def _postprocess_y_to_fix_miscount(self, y):
        """post process to manually fix head edu counting issue"""
        elements = y.split()
        head = elements.pop(0)
        headind = int(head[:-1].split('[edu')[1])
        if headind != self.edu_map:
            self.miss_count[self.docid].append([self.edu_map, headind]) # [real, pred headind]
            returny = f"[edu{self.edu_map}] " + ' '.join(elements)
            return returny
        return y
    
    # main looper function, start from edu 0, prepare input string at each step and send to prediction
    def extend(self):
        """add up annotation in each step to context"""
        while self.edu_map < self.max_edu_map - 1:
            self.edu_map += 1
            self.edu = self.edu_context[self.edu_map]
            if self.structure_type == 'focus':
                newinput = self.get_focus_input_annotation()
            elif self.structure_type == 'natural2':
                newinput = self.get_natural2_input_annotation()
            self.input_annotation = self.prefix + newinput
            self.input_annotation_context.append(newinput)
                
            encoded_str = self.encode(self.input_annotation)
            y = self.predict(encoded_str)
            
            self.prediction_str[self.edu_map] = y

            y1 = self._postprocess_focus_y_for_input_annotation(y)
            
            if self.structure_type == 'focus':
                self.annotation = self.input_annotation_context[-1] + ' | ' + y1
            elif self.structure_type == 'natural2':
                self.annotation = self.input_annotation_context[-1] + ' ' + y1.strip()
            self.annotation_context.append(self.annotation)
        
        assert self.edu_map+1 == len(self.edu_map_context) == len(self.annotation_context)
        self.done = True
        print(f"Failed parse: {self.fail_parse}")

def create_documents(document: str, dataset: str):
    """Read a json file and create data structure for raw input

    Args:
        document (str): input json file path
        dataset (str): name of the dataset, choose from stac and molweni
    
    Returns:
        list of dicts, each dict is one document with edus, edu id, sentences, speakers
    """
    assert os.path.exists(document), f"Document path {document} does not exist."
    
    input_documents = []
    
    if dataset == 'stac':
        with open(document, 'r') as inf:
            docs = inf.readlines()
    elif dataset == 'molweni':
        with open(document, 'r') as inf:
            docs = json.load(inf)
    
    # start process a document  
    for i, line in enumerate(docs):
        if dataset == 'stac':
            doc = json.loads(line)
        else:
            doc = line
            
        input_doc = {
            'id': -1,
            'edu_maps': [], #list of int, edu index
            'speakers': [],
            'edus': [], # list of str in the form of "speaker: sentence"
            'relations': [] # list of triplets: [(int_x,int_y,str_rel), () ...]
        }
        
        input_doc['id'] = doc['id']
        for i, edu in enumerate(doc['edus']):
            input_doc['speakers'].append(edu['speaker'])
            if dataset == 'stac': 
                assert i == edu['speechturn'] # only stac file has this attribute
                input_doc['edu_maps'].append(edu['speechturn'])
            elif dataset == 'molweni':
                input_doc['edu_maps'].append(i)
            input_doc['edus'].append(f"{edu['speaker']}: {edu['text']}")
        for rel in doc['relations']:
            input_doc['relations'].append((rel['x'], rel['y'], rel['type']))
        # end one document
        input_documents.append(input_doc)
    
    return input_documents


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()            
    
    parser.add_argument("--train_corpus", type=str, default="stac", help="train corpus: stac, molweni")
    parser.add_argument("--test_corpus", type=str, default="stac", help="test corpus: stac, molweni")
    parser.add_argument("-s", "--structure_type", type=str, default=None, required=True, \
                        help="transition-based: 'focus', 'natural2'.")
    parser.add_argument("-t", "--t5_family", type=str, default="t0-3b", help="choose from: 't0-3b', 'flan-t5', 't5'")  
    parser.add_argument("-m", "--model_size", type=str, default="3b", \
                        help="choose from: flan-t5: 'base', 'large', 'xl' 3B, 'xxl' 11B | t0: 3b, 11b, pp | t5: 3b, large")
    parser.add_argument("-b", "--bfloat16", action="store_true", default=False, help="if use brain float16, default=False")  
    parser.add_argument("-l", "--lr", type=str, default='5e-5', help="5e-5 up to xl/3b")  
    parser.add_argument("--seed", type=int, default=27, help="seed: 27, 16, etc")
    args = parser.parse_args()

    train_corpus = args.train_corpus
    test_corpus = args.test_corpus
    t5_family = args.t5_family 
    model_size = args.model_size
    structure_type = args.structure_type
    lr = args.lr
    bfloat16 = args.bfloat16
    seed = args.seed
    
    MAX_EDU_LEN = 37 # stac: 37, molweni: 14
 
    set_seed(seed=seed)

    # pretrained model
    fn_model_name = f"{t5_family}-{model_size}_train_{train_corpus}_{structure_type}_seed{seed}_{lr}"
    model_dir = os.path.join(FT_MODEL_DIR, f"{t5_family}-{model_size}_train_{train_corpus}_{structure_type}_seed{seed}_{lr}")
    
    # load test file, transition-based use original test file as input, e2e use processed structured test file
    testf = os.path.join(ROOT_DIR, f"data/{test_corpus}/test.json")
    
    # initialize test documents
    input_documents = create_documents(testf, test_corpus)
    
    # prediction starts
    total_time = time.time()
    total_results = 0
    all_predictions = {}
    
    for input_doc in input_documents:   
        t = time.time()    

        doc_state = State(input_doc, structure_type=structure_type, model_dir=model_dir, \
                        fn_model_name=fn_model_name, slide_window=True, max_len_doc=18, \
                        fix_count=False, bfloat16=bfloat16)
        if not doc_state.done:
            doc_state.extend()
        
        doc_prediction = doc_state.prediction_str
        doc_id = doc_state.docid
        for id, pred in doc_prediction.items():
            all_predictions[f"{doc_id}" + '_{:0>2d}'.format(id)] = pred
        
        print(
            f'time {time.time()-t}, round time/seq : {(time.time()-t)/len(doc_prediction)} '
            f'total time/seq: {(time.time()-total_time)/len(all_predictions)}'
            )
    # /END of iterative prediction    
    
    # write down prediction
    outfile_name = f"{t5_family}-{model_size}_train_{train_corpus}_test_{test_corpus}_transitionbase_{structure_type}_seed{seed}_gen512_lr{args.lr}_iterinfer.jsonl"
    
    res_file = os.path.join(ROOT_DIR, f"generation/{outfile_name}")
    print(f"writing result in {res_file}")
    
    with open(res_file, 'w') as of:
        for k, v in all_predictions.items():
            result = {'id': k, "gen_output": v}
            json.dump(result, of)
            of.write('\n')