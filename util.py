import numpy as np

## evaluation methods

def get_policy_direct(p,I,A,policy_info):
    Na = len(A)
    if p == 0:
        return dict([(a,1/Na) for a in A])
    elif I not in policy_info:
        return dict([(a,1/Na) for a in A])
    else:
        pa = policy_info[I]
        p_sum = sum([pa.get(a,0) for a in A])
        if p_sum == 0:
            return dict([(a,1/Na) for a in A])
        else:
            return dict([(a,max(pa.get(a,0),0)/p_sum) for a in A])

def V_policy(h,players,policy_info):
    if h.is_terminated():
        return h.evaluation()
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    ## recursive search
    probs = get_policy_direct(p,I,A,policy_info)
    value = 0
    for a in A:
        h_new = h.copy()
        h_new.transition(a)
        value_a = V_policy(h_new,players,policy_info) 
        value += probs[a] * value_a ## \sum_a p(a|I) v(h,a)
        
    return value

def V_BR(h,players,p_curr,policy_info):
    if h.is_terminated():
        return h.evaluation()
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    if p != p_curr:
        ## recursive search
        probs = get_policy_direct(p,I,A,policy_info)
        value = 0
        for a in A:
            h_new = h.copy()
            h_new.transition(a)
            value_a = V_BR(h_new,players,p_curr,policy_info) 
            value += probs[a] * value_a ## \sum_a p(a|I) v(h,a)
    else:
        if p_curr == 1:
            value = -1e8
            for a in A:
                h_new = h.copy()
                h_new.transition(a)
                value_a = V_BR(h_new,players,p_curr,policy_info)
                value = max(value,value_a) ## max v(h,a)
        else:
            value = 1e8
            for a in A:
                h_new = h.copy()
                h_new.transition(a)
                value_a = V_BR(h_new,players,p_curr,policy_info)
                value = min(value,value_a) ## min v(h,a)   
        
    return value

def eval_policy(h0,players,policy_info):
    return V_policy(h0,players,policy_info)

def eval_BR(h0,players,p_curr,policy_info):
    return V_BR(h0,players,p_curr,policy_info)

