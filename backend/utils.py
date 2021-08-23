import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    EQU = get_EQU_entity(tag_seq, char_seq)
    return PER, LOC, ORG, EQU


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    per='0'
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):      
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
             
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':            
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per='0'
            continue
    num_0=PER.count('0')
    for num in range(num_0):
        PER.remove('0')
    return PER


def get_LOC_entity(tag_seq, char_seq):  
    length = len(char_seq)
    LOC = []
    loc='0'    
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):    
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':           
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc='0' 
            continue
        
    num_0=LOC.count('0')
    for num in range(num_0):
        LOC.remove('0')       
    return LOC


def get_ORG_entity(tag_seq, char_seq):   
    length = len(char_seq)
    ORG = [] 
    org='0' 
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):       
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':         
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org='0'
            continue
        
    num_0=ORG.count('0')
    for num in range(num_0):
        ORG.remove('0')
    return ORG


def get_EQU_entity(tag_seq, char_seq):  
    length = len(char_seq)
    EQU = []  
    equ='0'
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)): 
        if tag == 'B-EQU':
            if 'equ' in locals().keys():
                EQU.append(equ)
                del equ
            equ = char
            if i+1 == length:
                EQU.append(equ)
        if tag == 'I-EQU': 
            equ += char
            if i+1 == length:
                EQU.append(equ)
        if tag not in ['I-EQU', 'B-EQU']:
            if 'equ' in locals().keys():
                EQU.append(equ)
                del equ 
            equ='0'
            continue  
    num_0=EQU.count('0')
    for num in range(num_0):
        EQU.remove('0')
    return EQU


#日志模板
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
