import numpy as np

def get_policy_from_regret_CFR(p,I,A,regret_info):
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
        
def CFR(h,players,p_curr,prob_info,regret_info,avg_policy_info,count_info):
    count_info['count'] += 1
    if h.is_terminated():
        return h.evaluation(p_curr)
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    ## recursive search
    probs = get_policy_from_regret_CFR(p,I,A,regret_info)
    values = {}
    value = 0
    for a in A:
        prob_a = probs[a]
        h_new = h.copy()
        h_new.transition(a)
        prob_info_new = {'all':prob_info['all'] * prob_a,
                         'i':prob_info['i'].copy(),
                         '-i':prob_info['-i'].copy(),
                        }
        prob_info_new['i'][p] *= prob_a
        for p_ in range(len(players)):
            if p_ != p:
                prob_info_new['-i'][p_] *= prob_a
        value_a = CFR(h_new,players,p_curr,prob_info_new,
                      regret_info,avg_policy_info,count_info) ## v(h,a)
        values[a] = value_a
        value += prob_a * value_a
    # print(h.action_history,values,probs,prob_info['-i'][p])
    
    ## update info-set meta info for the current player: 
    ## R(I,a) = \sum_h \sigma_{-i}(h) \cdot R(h,a), R(h,a) = v(h,a) - v(h)
    ## NOTE: v(h) depends on p!
    if p == p_curr:
        for info in [regret_info,avg_policy_info]:
            if I not in info:
                info[I] = dict()
        for a in A:
            regret_info[I][a] = regret_info[I].get(a,0) + prob_info['-i'][p] * (values[a] - value)
            avg_policy_info[I][a] = avg_policy_info[I].get(a,0) + prob_info['i'][p] * probs[a]
        
    return value

def train_CFR(h0,players,ps,regret_info,avg_policy_info,count_info,**kwargs):
    for p_curr in ps:
        prob_info = {'all':1,'i':np.ones((len(players),)),'-i':np.ones((len(players),))}
        CFR(h0.copy(),players,p_curr,prob_info,regret_info,avg_policy_info,count_info)

