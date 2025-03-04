import os
import json
from collections import defaultdict
import jellyfish
import copy
import argparse

from constant import *


def evaluate_gen_result(fted_model, train_corpus='stac', test_corpus='stac', \
                        structure_type='natural', max_infer_len=512, seed=27, lr='5e-5',\
                        count_root=True, SHOW_raw=True, SHOW_postprocess=True):
    """Evaluate end2end generation"""
    
    genf = f"generation/{fted_model}_train_{train_corpus}_test_{test_corpus}_{structure_type}_seed{seed}_gen{max_infer_len}_lr{lr}.jsonl"
    goldf = f"data/{test_corpus}_{structure_type}_test.json"
           
    # read predictions
    predictions = []
    with open(os.path.join(ROOT_DIR, genf), 'r') as inf:
        lines = inf.readlines()
        for i, l in enumerate(lines):
            predictions.append(json.loads(l)['gen_output'])
    
    # read gold
    golds = []    
    ids = []
    with open(os.path.join(ROOT_DIR, goldf), 'r') as inf:
        lines = inf.readlines()
        for i, l in enumerate(lines):
            golds.append(json.loads(l)['structure'])
            ids.append(json.loads(l)['id'])

    TP, TP_link = 0, 0
    FP, FP_link = 0, 0
    P, P_link = 0, 0 #raw
    clean_TP, clean_TP_link = 0, 0
    clean_FP, clean_FP_link = 0, 0
    clean_P, clean_P_link = 0, 0 #w post process
    G, G_link = 0, 0
    gold_pred_result = defaultdict(dict)
    gold_pred_result_post = defaultdict(dict)
    failed = defaultdict(list) #record failed parse generation
    hallucinate = defaultdict(list) #record imagnied edus in prediction
    
    # parse generation
    for i, (idd, g, p) in enumerate(zip(ids, golds, predictions)): #g, p is a dialogue
        gold_pred_result[idd]['gold'] = []
        gold_pred_result[idd]['pred'] = []
        gold_pred_result_post[idd]['gold'] = []
        gold_pred_result_post[idd]['pred'] = []
        
        # 1/ natural, labelmasked format
        if structure_type in ['natural', 'labelmasked']:
            # build gold triplets
            gold_rel = [gg.strip() for gg in g.split(';')]
            g_triplets = []
            max_g_edu = int(gold_rel[-1].split()[0][1:-1].split('edu')[1])
            if count_root:
                g_triplets.append(('[edu0]', 'root', '[edu0]'))
            for gr in gold_rel[1:]: #ignore the first relation "[edu0] is root"
                elements = gr.replace('is', ' ').replace('of', ' ').split() # eg: ['[edu7]', 'Acknowledgement', '[edu5]', 'Acknowledgement', '[edu3]', 'Acknowledgement']
                head = elements.pop(0) # the first element in element list is the head
                if len(elements) % 2 == 0:
                    for j in range(0, len(elements), 2):
                        g_triplets.append((head, elements[j], elements[j+1]))
                else:
                    print(gr)
            G += len(set(g_triplets))
            G_link += len(set(['-'.join([trip[0], trip[2]]) for trip in g_triplets])) #{'[edu1]-[edu0]', '[edu7]-[edu2]'}
            gold_pred_result[idd]['gold'] = g_triplets
            gold_pred_result_post[idd]['gold'] = g_triplets
            
            # build predicted triplets
            pred_rel = [pp.strip() for pp in p.split(';')]
            p_triplets = []
            if count_root and pred_rel[0] == '[edu0] is root':
                p_triplets.append(('[edu0]', 'root', '[edu0]'))
            for pr in pred_rel[1:]:
                elements = pr.replace('is', ' ').replace('of', ' ').split() # eg: ['[edu7]', 'Acknowledgement', '[edu5]', 'Acknowledgement', '[edu3]', 'Acknowledgement']
                if len(elements) > 0:
                    head = elements.pop(0) # the first element in element list is the head
                    if len(elements) % 2 == 0:
                        for j in range(0, len(elements), 2):
                            p_triplets.append((head, elements[j], elements[j+1]))
                    else:
                        print(pr)
            P += len(set(p_triplets)) #--> clean_P
            P_link += len(set(['-'.join([trip[0], trip[2]]) for trip in p_triplets])) #--> clean_P_link
            # postprocess ignore over-predicted edus
            p_triplets = list(dict.fromkeys(p_triplets)) #post1: remove duplicate while keep order
            clean_p_triplets = []
            miss_edu = False
            under_len = -1
            for ip, trip in enumerate(p_triplets):
                head_id = int(trip[0][1:-1].split('edu')[1])
                if head_id <= max_g_edu: #post2: edu length constraint
                    clean_p_triplets.append(trip)
                    clean_P_link += 1
                else:
                    hallucinate[idd].append(head_id)
                if ip == len(p_triplets)-1 and head_id < max_g_edu:
                    miss_edu = True
                    under_len = max_g_edu - head_id
            clean_P += len(clean_p_triplets)
            # post3: add missing edu link with nearest neighbour and default relation qap
            if miss_edu:
                for im in range(under_len):
                    failed[idd].append(f"[edu{head_id+1}]")
                    if structure_type == 'natural':
                        clean_p_triplets.append((f"[edu{head_id+1}]", DEFAULT_REL, f"[edu{head_id}]"))
                    if structure_type == 'labelmasked':
                        clean_p_triplets.append((f"[edu{head_id+1}]", DEFAULT_RELMASK, f"[edu{head_id}]"))
                    clean_P_link += 1
                    head_id += 1
            if SHOW_raw:
                gold_pred_result[idd]['pred'] = p_triplets
            if SHOW_postprocess:
                gold_pred_result_post[idd]['pred'] = clean_p_triplets
        # 1/ end 
                
        # 2. augmented format
        elif structure_type == 'augmented':
            # build gold triplets
            gold_rel = [gg.strip() for gg in g.split('] [')]
            gold_rel[0] = gold_rel[0].strip('[ ')
            gold_rel[-1] = gold_rel[-1].strip(' ]')
            max_g_edu = len(gold_rel) - 1 #delete the first relation root=edu0
            g_quadruple = []
            g_triplets = []
            if count_root:
                first_edu = [ele.strip() for ele in gold_rel[0].split('|')]
                g_quadruple.append((first_edu[0], 'edu0', 'root', 'edu0'))
            for gr in gold_rel[1:]:
                elements = [ele.strip() for ele in gr.split('|')] # eg: ['[edu7]', 'Acknowledgement', '[edu5]', 'Acknowledgement', '[edu3]', 'Acknowledgement']
                if len(elements) != 3:
                    print(i, elements)
                else:
                    headtxt = elements.pop(0) 
                    headidx = elements.pop(0)
                    deprel = elements[0].replace('=', '').split()
                    if len(deprel) % 2 == 0:
                        for j in range(0, len(deprel), 2):
                            g_quadruple.append((headtxt, headidx, deprel[j], deprel[j+1]))
                    else:
                        print(gr)
            G += len(set(g_quadruple))
            G_link += len(set(['-'.join([trip[1], trip[3]]) for trip in g_quadruple])) #{'[edu1]-[edu0]', '[edu7]-[edu2]'}
            g_triplets = [qua[1:] for qua in g_quadruple]
            gold_pred_result[idd]['gold'] = g_triplets
            gold_pred_result_post[idd]['gold'] = g_triplets
        
            # parse generated output
            pred_rel = [pp.strip() for pp in p.split('] [')]
            pred_rel[0] = pred_rel[0].strip('[ ')
            pred_rel[-1] = pred_rel[-1].strip(' ]')
            
            p_triplets = []
            clean_p_triplets = []
            g_quadruple_dupli = copy.deepcopy(g_quadruple)
            p_rel_dupli = copy.deepcopy(pred_rel)
            for gg, (gtxt, gidx, rr, dd) in enumerate(g_quadruple): 
                gg_dupli = g_quadruple_dupli.index((gtxt, gidx, rr, dd))
                for pp, pr in enumerate(p_rel_dupli):
                    elements = [ele.strip() for ele in pr.split('|')]
                    if len(elements) != 3:
                        print(i, elements)
                    else:
                        headtxt = elements.pop(0) 
                        headidx = elements.pop(0)  
                        # exact match
                        if jellyfish.jaro_similarity(headtxt, gtxt) > 0.96 and headidx == gidx: # heuristic: 0.96 can best cover space diff in generation and gold
                            deprel = elements[0].replace('=', '').split()
                            if len(deprel) % 2 == 0:
                                for j in range(0, len(deprel), 2):
                                    p_triplets.append((headidx, deprel[j], deprel[j+1]))
                                    clean_p_triplets.append((headidx, deprel[j], deprel[j+1]))
                                del g_quadruple_dupli[gg_dupli]
                                if pp < len(p_rel_dupli):
                                    del p_rel_dupli[pp]
                            break
                        elif headtxt == gtxt and headidx != gidx: #post: wrong predict edu index, correct it, also correct dependent edu index
                            gap = int(gidx.split('edu')[1]) - int(headidx.split('edu')[1])
                            corrected_idx = gidx
                            deprel = elements[0].replace('=', '').split()
                            if len(deprel) % 2 == 0:
                                for j in range(0, len(deprel), 2):
                                    corrected_depidx = f"edu{int(deprel[j+1].split('edu')[1])+gap}"
                                    if (corrected_idx, deprel[j], corrected_depidx) not in clean_p_triplets:
                                        clean_p_triplets.append((corrected_idx, deprel[j], corrected_depidx))
                                del g_quadruple_dupli[gg_dupli]
                                if pp < len(p_rel_dupli):
                                    del p_rel_dupli[pp]
                            break
                        elif headidx == gidx: #post:fail predict edu txt, but correct edu idx, mostly this case
                            deprel = elements[0].replace('=', '').split()
                            if len(deprel) % 2 == 0:
                                for j in range(0, len(deprel), 2):
                                    p_triplets.append((headidx, deprel[j], deprel[j+1]))
                                    if (headidx, deprel[j], deprel[j+1]) not in clean_p_triplets:
                                        clean_p_triplets.append((headidx, deprel[j], deprel[j+1]))
                                del g_quadruple_dupli[gg_dupli]
                                if pp < len(p_rel_dupli):
                                    del p_rel_dupli[pp]
                            break
                        else: # fail predict txt, fail predict index, can't locate
                            pass

            if p_rel_dupli != []:
                failed[idd].extend(p_rel_dupli)
                                
            P += len(set(p_triplets)) #--> clean_P
            P_link += len(set(['-'.join([trip[0], trip[2]]) for trip in p_triplets])) #--> clean_P_link
            # post: add missing edu link with nearest neighbour and default relation qap
            miss_edu = False
            under_len = -1
            complet_edu_rg = [e[0] for e in g_triplets]
            pred_edu_rg = [e[0] for e in clean_p_triplets]
            
            if set(complet_edu_rg) - set(pred_edu_rg) != set(): #sth in gold set is not in pred set
                miss_edu = True
            if miss_edu:
                clean_p_triplets_new = []
                for cand_e in complet_edu_rg:
                    if cand_e in pred_edu_rg:
                        clean_p_triplets_new.extend([t for t in clean_p_triplets if t[0] == cand_e])
                    else:
                        clean_p_triplets_new.append((cand_e, DEFAULT_REL, f"edu{int(cand_e.split('edu')[1])-1}"))
                clean_p_triplets = clean_p_triplets_new
            clean_P += len(clean_p_triplets)
            clean_P_link += len(set(['-'.join([trip[0], trip[2]]) for trip in clean_p_triplets]))
            
            if SHOW_raw:
                gold_pred_result[idd]['pred'] = p_triplets
            if SHOW_postprocess:
                gold_pred_result_post[idd]['pred'] = clean_p_triplets
        # 2/ end 
        
        # link+rel
        TP += len(set(p_triplets).intersection(set(g_triplets)))
        FP += len(set(p_triplets) - set(g_triplets)) 
        clean_TP += len(set(clean_p_triplets).intersection(set(g_triplets)))
        clean_FP += len(set(clean_p_triplets) - set(g_triplets)) 

        # only link
        TP_link += len(set(['-'.join([trip[0], trip[2]]) for trip in p_triplets]).intersection(set(['-'.join([trip[0], trip[2]]) for trip in g_triplets])))
        FP_link += len(set(['-'.join([trip[0], trip[2]]) for trip in p_triplets]) - (set(['-'.join([trip[0], trip[2]]) for trip in g_triplets]))) 
        clean_TP_link += len(set(['-'.join([trip[0], trip[2]]) for trip in clean_p_triplets]).intersection(set(['-'.join([trip[0], trip[2]]) for trip in g_triplets])))
        clean_FP_link += len(set(['-'.join([trip[0], trip[2]]) for trip in clean_p_triplets]) - (set(['-'.join([trip[0], trip[2]]) for trip in g_triplets])))

    
    print(f"====\n{test_corpus} test set, {structure_type}, seed{seed}\n====")
    print(f"[{structure_type}]")
    if SHOW_raw:
        recall = TP / G * 100
        precision = TP / (P-4) * 100 #in gold, docs in line 2,78,81,91 miss 1 edge
        f1 = 2 * recall * precision / (recall + precision)
        print(f"Raw  [link+rel] recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1: {round(f1, 2)}") 
        recall = TP_link / G_link * 100
        precision = TP_link / (P_link-4) * 100
        f1 = 2 * recall * precision / (recall + precision)
        print(f"Raw  [linkonly] recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1: {round(f1, 2)}") 
    if SHOW_postprocess:
        recall = clean_TP / G * 100
        precision = clean_TP / (clean_P-4) * 100
        f1 = 2 * recall * precision / (recall + precision)
        print(f"Post [link+rel] recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1: {round(f1, 2)}") 
        recall = clean_TP_link / G_link * 100
        precision = clean_TP_link / (clean_P_link-4) * 100
        f1 = 2 * recall * precision / (recall + precision)
        print(f"Post [linkonly] recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1: {round(f1, 2)}") 
    print()


def evaluate_transition_result(fted_model, train_corpus='stac', test_corpus='stac', structure_type='natural2',\
                            max_infer_len=512, seed=27, lr='5e-5', count_root=True):
    """Evaluate transition-based generation"""
    
    genf = f"generation/{fted_model}_train_{train_corpus}_test_{test_corpus}_transitionbase_{structure_type}_seed{seed}_gen{max_infer_len}_lr{lr}_iterinfer.jsonl"
    goldf = f"data/{test_corpus}_{structure_type}_test.json"
        
    # read predictions
    predictions = []
    with open(os.path.join(ROOT_DIR, genf), 'r') as inf:
        lines = inf.readlines()
        for i, l in enumerate(lines):
            predictions.append(json.loads(l)['gen_output'])
    
    # read gold
    golds = []    
    ids = []
    with open(os.path.join(ROOT_DIR, goldf), 'r') as inf:
        lines = inf.readlines()
        for i, l in enumerate(lines):
            golds.append(json.loads(l)['structure'])
            ids.append(json.loads(l)['id'])

    TP, TP_link = 0, 0
    FP, FP_link = 0, 0
    P, P_link = 0, 0
    G, G_link = 0, 0
    gold_pred_result = defaultdict(dict)
    failed = defaultdict(list) #record failed parse generation
    
    # parse generation
    for _, (idd, g, p) in enumerate(zip(ids, golds, predictions)): #g, p is a utterance
        headedu_idd = int(idd.split('_')[-1])
        headedu_str = f"[edu{headedu_idd}]"
        doc_idd = str(idd.rsplit('_', 1)[0])
        if doc_idd not in gold_pred_result.keys():
            gold_pred_result[doc_idd]['gold'] = []
            gold_pred_result[doc_idd]['pred'] = []
            failed[doc_idd] = [] # record repetitive prediction
        
        # build gold triplets
        g_ele = []
        g_lin = []
        if count_root and g.strip() == 'root':
            g_ele.append('root')
            g_lin.append('[edu0]')
            G += 1
            G_link += 1
            gold_pred_result[doc_idd]['gold'].append(('[edu0]', 'root', '[edu0]'))
        if g.strip() != "root":
            elements = g.replace('is', ' ').replace('of', ' ').split()
            if len(elements) % 2 == 0:
                for j in range(0, len(elements), 2):                      
                    g_ele.append((elements[j], elements[j+1]))
                    g_lin.append(elements[j+1])
                    G += 1
                    G_link += 1
                    gold_pred_result[doc_idd]['gold'].append((headedu_str, elements[j], elements[j+1]))

        p_ele = []
        p_lin = []
        if count_root and p.strip() == 'root':
            p_ele.append('root')
            p_lin.append('[edu0]')
            P += 1
            P_link += 1
            gold_pred_result[doc_idd]['pred'].append(('[edu0]', 'root', '[edu0]'))
        if p.strip() != "root":
            elements = p.replace('is', ' ').replace('of', ' ').split()
            if len(elements) % 2 == 0:
                for j in range(0, len(elements), 2):
                    if (elements[j], elements[j+1]) not in p_ele:
                        p_ele.append((elements[j], elements[j+1]))
                        gold_pred_result[doc_idd]['pred'].append((headedu_str, elements[j], elements[j+1]))
                        P += 1
                    else:
                        failed[doc_idd].append((headedu_str, elements[j], elements[j+1])) #repetitive prediction
                    if elements[j+1] not in p_lin:
                        p_lin.append(elements[j+1])
                        P_link += 1
            
        TP += len(set(p_ele).intersection(set(g_ele)))
        FP += len(set(p_ele) - set(g_ele))
        TP_link += len(set(p_lin).intersection(set(g_lin)))
        FP_link += len(set(p_lin) - set(g_lin))
                       
    recall = TP / G * 100
    precision = TP / P * 100
    f1 = 2 * recall * precision / (recall + precision)
    print(f"[link+rel] recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1: {round(f1, 2)}") 
    recall = TP_link / G_link * 100
    precision = TP_link / (P_link-4) * 100
    f1 = 2 * recall * precision / (recall + precision)
    print(f"[linkonly] recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1: {round(f1, 2)}") 

           
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()            
    
    parser.add_argument("--fted_model", type=str, help="fine-tuned model, e.g., 't0-3b'")
    parser.add_argument("--train_corpus", type=str, default="stac", help="train corpus: stac, molweni")
    parser.add_argument("--test_corpus", type=str, default="stac", help="test corpus: stac, molweni")
    parser.add_argument("-s", "--structure_type", type=str, default=None, required=True, \
                        help="end2end: 'natural', 'augmented', 'labelmasked' | transition-based: 'focus', 'natural2'.")
    parser.add_argument("-l", "--lr", type=str, default='5e-5', help="5e-5 up to xl/3b")  
    parser.add_argument("--seed", type=int, default=27, help="seed: 27, 16, etc")
    args = parser.parse_args()

    fted_model = args.fted_model
    train_corpus = args.train_corpus
    test_corpus = args.test_corpus
    structure_type = args.structure_type
    lr = args.lr
    seed = args.seed

    if structure_type == 'augmented':
        max_infer_len=1024
    else:
        max_infer_len=512
        
    evaluate_gen_result(fted_model, train_corpus=train_corpus, test_corpus=test_corpus, \
                        structure_type=structure_type, max_infer_len=max_infer_len, seed=seed, lr=lr)
    

    evaluate_transition_result(fted_model, train_corpus=train_corpus, test_corpus=test_corpus, \
                            structure_type=structure_type, max_infer_len=max_infer_len, seed=seed, lr=lr)
    