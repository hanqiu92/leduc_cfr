import numpy as np
import torch
import matplotlib.pyplot as plt

class Memory:
    def __init__(self,process_func,memory_size=1e6):
        self.memory_size = int(memory_size)
        self.process_func = process_func
        self.data_size = 0
        self.data_size_added = 0
        self.data = []

    def add_memory(self,I,A,ys,t):
        for a in A:
            data_point = (I,a,self.process_func(I,a),ys[a],t)
            self.data_size_added += 1
            if self.data_size < self.memory_size:
                self.data.append(data_point)
                self.data_size += 1
            else:
                idx = np.random.randint(self.data_size_added)
                if idx < self.data_size:
                    self.data[idx] = data_point

    def take_batch(self,batch_size=1024):
        if self.data_size < batch_size:
            batch_idxs = np.arange(self.data_size)
        else:
            batch_idxs = np.random.choice(self.data_size,batch_size,replace=False)
        return [self.data[idx] for idx in batch_idxs]

class Model:
    def __init__(self,network,process_func,name='',lr=1e-3,batch_size=1024,if_display=False,display_params=(0.025,1)):
        self.network = network
        self.lr = lr
        self.reset_optimizer()
        self.batch_size = batch_size
        self.process_func = process_func
        self.name = name
        self.if_display = if_display
        self.display_params = display_params

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=self.lr)

    def train(self,memory,N_iters=1,reset_optimizer_flag=True):
        if reset_optimizer_flag:
            self.network.reset_parameters()
            self.reset_optimizer()

        if self.if_display:
            losses = []
        for iter_ in range(N_iters):
            batch_data = memory.take_batch(self.batch_size)
            _,_,batch_x,batch_y,batch_w = zip(*batch_data)
            y_torch = torch.tensor(batch_y).float()
            w_torch = torch.tensor(batch_w).float()
            zero_torch = torch.tensor(0).float()

            out = self.network(torch.tensor(batch_x).float())[:,0]
            loss1 = torch.mean(w_torch * (out - y_torch) ** 2) / torch.mean(w_torch)
            # loss2 = torch.mean(w_torch * (torch.maximum(out,zero_torch) - torch.maximum(y_torch,zero_torch)) ** 2) / torch.mean(w_torch)
            # loss = loss1 / 2 + loss2 / 2
            loss = loss1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.if_display:
                losses.append(loss.item())

        if self.if_display:
            alpha,start_idx = self.display_params
            plt.figure()

            losses_smooth = [losses[0]]
            for l in losses[1:]:
                losses_smooth.append(losses_smooth[-1] * (1-alpha) + alpha * l)
            plt.plot(losses_smooth[start_idx:],'b')

            plt.title(self.name)
            plt.grid()
            plt.show()

    def predict(self,I,A):
        batch_x = [self.process_func(I,a) for a in A]
        out = self.network(torch.tensor(batch_x).float()).detach().numpy()[:,0]
        out_map = dict(zip(A,out))
        return out_map

class MultiModel(Model):
    def __init__(self,n_models,network_init_func,process_func,name='',
                 lr=1e-3,batch_size=1024,if_display=False,display_params=(0.025,1)):
        self.models = []
        for _ in range(n_models):
            self.models.append(
                Model(network_init_func(),process_func,name,lr,batch_size,if_display,display_params)
            )

    def reset_optimizer(self):
        for model in self.models:
            model.reset_optimizer()

    def train(self,memory,N_iters=1,reset_optimizer_flag=True):
        for model in self.models:
            model.train(memory,N_iters,reset_optimizer_flag)

    def predict(self,I,A):
        outputs = []
        for model in self.models:
            outputs.append(model.predict(I,A))
        out_map = {}
        for a in A:
            preds = np.array([out.get(a,0) for out in outputs])
            # out_map[a] = np.max(preds)
            out_map[a] = np.mean(preds) + 1.25 * np.std(preds)
        return out_map

def get_policy_from_regret_deep_CFR(p,I,A,regret_models):
    ## regret matching
    Na = len(A)
    if p == 0:
        return dict([(a,1/Na) for a in A])
    else:
        Ra = regret_models[p].predict(I,A)
        R_sum = sum([max(Ra.get(a,0),0) for a in A])
        if R_sum == 0:
            return dict([(a,1/Na) for a in A])
        else:
            return dict([(a,max(Ra.get(a,0),0)/R_sum) for a in A])
        
def deep_CFR(h,players,p_curr,prob_info,regret_models,regret_memory,policy_memory,iter_count,count_info):
    count_info['count'] += 1
    if h.is_terminated():
        return h.evaluation(p_curr)
    
    p = h.get_player()
    I = h.get_infoset()
    
    P = players[p]
    A = P.get_action_set(I)

    probs = get_policy_from_regret_deep_CFR(p,I,A,regret_models)
    if p == p_curr:
        ## recursive search only for the curr player
        values = {}
        value = 0
        for a in A:
            prob_a = probs[a]
            h_new = h.copy()
            h_new.transition(a)
            value_a = deep_CFR(h_new,players,p_curr,prob_info,
                        regret_models,regret_memory,policy_memory,
                        iter_count,count_info) ## v(h,a)
            values[a] = value_a
            value += prob_a * value_a
        ## update regret memory
        if p != 0:
            regrets = dict([(a,values[a] - value) for a in A])
            regret_memory[p].add_memory(I,A,regrets,iter_count)
        return value
    else:
        ## update policy memory
        if p != 0:
            policy_memory.add_memory(I,A,probs,iter_count)
        probs_arr = [probs[a] for a in A]
        a_idx = np.random.choice(len(probs_arr),p=probs_arr)
        a = A[a_idx]
        prob_a = probs[a]
        h_new = h.copy()
        h_new.transition(a)
        value = deep_CFR(h_new,players,p_curr,prob_info,
                    regret_models,regret_memory,policy_memory,
                        iter_count,count_info) ## v(h,a)
        return value

    return value

def train_deep_CFR(h0,players,ps,regret_models,regret_memory,policy_memory,count_info,**kwargs):
    inner_iter = kwargs.get('inner_iter',10)
    regret_train_iter = kwargs.get('regret_train_iter',1000)
    iter_count = kwargs.get('curr_iter',0) + 1
    for p_curr in ps:
        for k in range(inner_iter):
            prob_info = {'all':1,'i':np.ones((len(players),)),'-i':np.ones((len(players),))}
            deep_CFR(h0.copy(),players,p_curr,prob_info,regret_models,regret_memory,policy_memory,iter_count,count_info)
        regret_models[p_curr].train(regret_memory[p_curr],N_iters=regret_train_iter,reset_optimizer_flag=True)
