# paths
ROOT_DIR = "yourpathto/seq2seq-disc-parsing"
DATA_DIR = "yourpathto/seq2seq-disc-parsing/data"
HF_MODEL_DIR = "yourpathto/.cache/huggingface/hub"
FT_MODEL_DIR = "yourpathto/seq2seq-disc-parsing/ft-models"

# default values
DEFAULT_REL = "Question_answer_pair"
DEFAULT_RELMASK = "rel0"

# relation labels
MASKLABEL = {'Question_answer_pair': 'rel0',
            'Comment':'rel1',
            'Acknowledgement':'rel2',
            'Continuation':'rel3',
            'Elaboration':'rel4',
            'Q_Elab':'rel5',
            'Result':'rel6',
            'Contrast':'rel7',
            'Explanation':'rel8',
            'Clarification_question':'rel9',
            'Parallel':'rel10',
            'Correction':'rel11',
            'Alternation':'rel12',
            'Narration':'rel13',
            'Conditional':'rel14',
            'Background':'rel15'}

LABEL2ID = {'Question_answer_pair':0,
            'Comment':1,
            'Acknowledgement':2,
            'Continuation':3,
            'Elaboration':4,
            'Q_Elab':5,
            'Result':6,
            'Contrast':7,
            'Explanation':8,
            'Clarification_question':9,
            'Parallel':10,
            'Correction':11,
            'Alternation':12,
            'Narration':13,
            'Conditional':14,
            'Background':15}

ID2LABEL = {0: 'Question_answer_pair',
            1: 'Comment',
            2: 'Acknowledgement',
            3: 'Continuation',
            4: 'Elaboration',
            5: 'Q_Elab',
            6: 'Result',
            7: 'Contrast',
            8: 'Explanation',
            9: 'Clarification_question',
            10: 'Parallel',
            11: 'Correction',
            12: 'Alternation',
            13: 'Narration',
            14: 'Conditional',
            15: 'Background'}

# stored model checkpoint
MODEL2CHECKPOINT = {'t0-3b_train_stac_focus_seed27_5e-5': 'checkpoint-6399',
                    't0-3b_train_molweni_focus_seed27_5e-5': 'checkpoint-19872'
                    }