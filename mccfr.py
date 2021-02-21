import numpy as np

def get_policy_from_regret_MCCFR(p,I,A,regret_info):
    ## regret matching
    Na = len(A)
    if p == 0:
        return dict([(a,1/Na) for a in A])
    elif I not in regret_info:
        return dict([(a,1/Na) for a in A])
    else:
        Ra = regret_info[I]
        R_sum = sum([max(Ra.get(a,0),0) for a in A])
        if R_sum == 0:
            return dict([(a,1/Na) for a in A])
        else:
            return dict([(a,max(Ra.get(a,0),0)/R_sum) for a in A])
        
def MCCFR(h,players,p_curr,prob_info,regret_info,avg_policy_info,count_info):
    count_info['count'] += 1
    if h.is_terminated():
        return h.evaluation(p_curr)
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    probs = get_policy_from_regret_MCCFR(p,I,A,regret_info)
    if p == p_curr:
        ## recursive search only for the curr player
        values = {}
        value = 0
        for a in A:
            prob_a = probs[a]
            h_new = h.copy()
            h_new.transition(a)
            value_a = MCCFR(h_new,players,p_curr,prob_info,
                        regret_info,avg_policy_info,count_info) ## v(h,a)
            values[a] = value_a
            value += prob_a * value_a
    else:
        probs_arr = [probs[a] for a in A]
        a_idx = np.random.choice(len(probs_arr),p=probs_arr)
        a = A[a_idx]
        prob_a = probs[a]
        h_new = h.copy()
        h_new.transition(a)
        value = MCCFR(h_new,players,p_curr,prob_info,
                    regret_info,avg_policy_info,count_info) ## v(h,a)
    
    ## update info-set meta info for the current player: 
    ## R(I,a) = R(h,a), R(h,a) = v(h,a) - v(h)
    ## NOTE: v(h) depends on p!
    if p == p_curr:
        if I not in regret_info:
            regret_info[I] = dict()
        for a in A:
            regret_info[I][a] = regret_info[I].get(a,0) + (values[a] - value)
    else:
        if I not in avg_policy_info:
            avg_policy_info[I] = dict()
        for a in A:
            avg_policy_info[I][a] = avg_policy_info[I].get(a,0) + probs[a]
        
    return value

def train_MCCFR(h0,players,ps,regret_info,avg_policy_info,count_info,**kwargs):
    for p_curr in ps:
        prob_info = {'all':1,'i':np.ones((len(players),)),'-i':np.ones((len(players),))}
        MCCFR(h0.copy(),players,p_curr,prob_info,regret_info,avg_policy_info,count_info)

def discount_avg_policy_MCCFR(avg_policy_info,iter_=1):
    discount = iter_ / (iter_ + 1)
    for policy in avg_policy_info.values():
        for a in policy.keys():
            policy[a] *= discount

def train_linear_MCCFR(h0,players,ps,regret_info,avg_policy_info,count_info,iter_=0,**kwargs):
    if iter_ > 0:
        discount_avg_policy_MCCFR(avg_policy_info,iter_=iter_)
    for p_curr in ps:
        prob_info = {'all':1,'i':np.ones((len(players),)),'-i':np.ones((len(players),))}
        MCCFR(h0.copy(),players,p_curr,prob_info,regret_info,avg_policy_info,count_info)
