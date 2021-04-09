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
        p_sum = sum([max(pa.get(a,0),0) for a in A])
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

    if p != p_curr or p == 0:
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


## evaluation methods, for models

def get_policy_direct_nn(p,I,A,policy_model):
    Na = len(A)
    if p == 0:
        return dict([(a,1/Na) for a in A])
    else:
        # pa = policy_info[I]
        pa = policy_model.predict(I,A)
        p_sum = sum([max(pa.get(a,0),0) for a in A])
        if p_sum == 0:
            return dict([(a,1/Na) for a in A])
        else:
            return dict([(a,max(pa.get(a,0),0)/p_sum) for a in A])

def V_policy_nn(h,players,policy_model):
    if h.is_terminated():
        return h.evaluation()
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    ## recursive search
    probs = get_policy_direct_nn(p,I,A,policy_model)
    value = 0
    for a in A:
        h_new = h.copy()
        h_new.transition(a)
        value_a = V_policy_nn(h_new,players,policy_model) 
        value += probs[a] * value_a ## \sum_a p(a|I) v(h,a)
        
    return value

def V_BR_nn(h,players,p_curr,policy_model):
    if h.is_terminated():
        return h.evaluation()
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    if p != p_curr or p == 0:
        ## recursive search
        probs = get_policy_direct_nn(p,I,A,policy_model)
        value = 0
        for a in A:
            h_new = h.copy()
            h_new.transition(a)
            value_a = V_BR_nn(h_new,players,p_curr,policy_model) 
            value += probs[a] * value_a ## \sum_a p(a|I) v(h,a)
    else:
        if p_curr == 1:
            value = -1e8
            for a in A:
                h_new = h.copy()
                h_new.transition(a)
                value_a = V_BR_nn(h_new,players,p_curr,policy_model)
                value = max(value,value_a) ## max v(h,a)
        else:
            value = 1e8
            for a in A:
                h_new = h.copy()
                h_new.transition(a)
                value_a = V_BR_nn(h_new,players,p_curr,policy_model)
                value = min(value,value_a) ## min v(h,a)   
        
    return value

def eval_policy_nn(h0,players,policy_model):
    return V_policy_nn(h0,players,policy_model)

def eval_BR_nn(h0,players,p_curr,policy_model):
    return V_BR_nn(h0,players,p_curr,policy_model)
