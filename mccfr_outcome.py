import numpy as np

def get_policy_from_regret_MCCFR_outcome(p,I,A,regret_info):
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
        
def MCCFR_outcome(h,players,p_curr,prob_info,regret_info,avg_policy_info,count_info,explore_rate=0.1):
    count_info['count'] += 1
    if h.is_terminated():
        return h.evaluation(p_curr)
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    probs = get_policy_from_regret_MCCFR_outcome(p,I,A,regret_info)
    probs_arr = [probs[a] for a in A]
    if p == p_curr:
        probs_arr_sample = (1 - explore_rate) * np.array(probs_arr) + explore_rate * np.ones((len(A),)) / len(A)
    else:
        probs_arr_sample = probs_arr
    a_idx = np.random.choice(len(probs_arr_sample),p=probs_arr_sample)
    a = A[a_idx]
    prob_a = probs[a]
    if p == p_curr:
        prob_a_sample = (1 - explore_rate) * prob_a + explore_rate / len(A)
    else:
        prob_a_sample = prob_a
    h_new = h.copy()
    h_new.transition(a)
    prob_info_new = {'all':prob_info['all'] * prob_a_sample,
                        'i':prob_info['i'].copy(),
                        '-i':prob_info['-i'].copy(),
                    }
    prob_info_new['i'][p] *= prob_a
    for p_ in range(len(players)):
        if p_ != p:
            prob_info_new['-i'][p_] *= prob_a
    value_a = MCCFR_outcome(h_new,players,p_curr,prob_info_new,
                regret_info,avg_policy_info,count_info) ## v(h,a)
    if p == p_curr:
        values = {}
        values[a] = value_a / prob_a_sample
        value = values[a] * prob_a
    else:
        value = value_a
    
    ## update info-set meta info for the current player: 
    ## R(I,a) = R(h,a), R(h,a) = v(h,a) - v(h)
    ## NOTE: v(h) depends on p!
    if p == p_curr:
        if I not in regret_info:
            regret_info[I] = dict()
        for a in A:
            regret_info[I][a] = regret_info[I].get(a,0) + prob_info['-i'][p] * (values.get(a,0) - value) / prob_info['all']
        if I not in avg_policy_info:
            avg_policy_info[I] = dict()
        for a in A:
            avg_policy_info[I][a] = avg_policy_info[I].get(a,0) + prob_info['i'][p] * probs[a] / prob_info['all']
        
    return value

def train_MCCFR_outcome(h0,players,ps,regret_info,avg_policy_info,count_info,**kwargs):
    explore_rate = kwargs.get('explore_rate',0.1)
    for p_curr in ps:
        prob_info = {'all':1,'i':np.ones((len(players),)),'-i':np.ones((len(players),))}
        MCCFR_outcome(h0.copy(),players,p_curr,prob_info,regret_info,avg_policy_info,count_info,explore_rate=explore_rate)

def discount_avg_policy_MCCFR_outcome(avg_policy_info,iter_=1):
    discount = iter_ / (iter_ + 1)
    for policy in avg_policy_info.values():
        for a in policy.keys():
            policy[a] *= discount

def train_linear_MCCFR_outcome(h0,players,ps,regret_info,avg_policy_info,count_info,iter_=0,**kwargs):
    explore_rate = kwargs.get('explore_rate',0.1)
    if iter_ > 0:
        discount_avg_policy_MCCFR_outcome(avg_policy_info,iter_=iter_)
    for p_curr in ps:
        prob_info = {'all':1,'i':np.ones((len(players),)),'-i':np.ones((len(players),))}
        MCCFR_outcome(h0.copy(),players,p_curr,prob_info,regret_info,avg_policy_info,count_info,explore_rate=explore_rate)
