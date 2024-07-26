# Dialogue Discourse Parsing as Generation: a Sequence-to-Sequence LLM-based Approach

This is the source code repository for the paper Dialogue Discourse Parsing as Generation: a Sequence-to-Sequence LLM-based Approach ([SIGDial 2024](https://2024.sigdial.org)).

<img src="./pic/seq2seq-disc-parse.png" alt="drawing" width="600"/>

## Datasets
### STAC
We used the **linguistic-only** STAC corpus and followed the separation of train, dev, test in [Shi and Huang, A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues. In AAAI, 2019](https://github.com/shizhouxing/DialogueDiscourseParsing).
The latest available verison on the website is available [here](https://www.irit.fr/STAC/corpus.html). 
We share the dataset we used in `data/stac/`.


### Molweni
Download from [here](https://github.com/HIT-SCIR/Molweni). We use the original separation of train, dev, and test.
Download the dataset and place it in `data/molweni/`. 

## How to run
Here is a step-by-step guide to fine-tune a T5 family model for discourse parsing:

### Create a virtual environment
```
$ source virtualenvname/bin/activate
$ cd Seq2Seq-DDP/
$ pip install -r requirements.txt
```

### Prepare structured data for fine-tuning

In `dataprocess.py`: process the original stac/molweni dataset and convert the raw text to structured text.
Choose the structured text from: 'natural', 'augmented' (Seq2Seq-DDP) and 'focus', 'natural2' (Seq2Seq-DDP+transition).
Examples for each structure type are given in `data/stac_{structure}_train.json`.

### Fine-tuning

In `train.py`: give "do_train" as argument. 
This code fine-tunes a t5 familiy model for discourse parsing. 

### End2end prediction and transition-based prediction

- Seq2seq-DDP prediction: in `train.py`, give argument "do_test", choose structure type from 'augmented', 'natural'.
Make sure to first put the fine-tuned model checkpoint in `constant.py`. Results will be written in `generation/`.

- Seq2Seq-DDP+transition system prediction: in `transition_predict.py`: choose structure type from 'focus', 'natural2'.

- `evaluate.py`: Evaluate predicted files in `generation/` and calculate scores.

- `constant.py`: store paths, labels, etc.

## Citation
Soon