import numpy as np

class ChancePlayer:
    def __init__(self,N_cards):
        self.N_cards = N_cards
        self.deck = sum([list(range(N_cards)) for _ in range(2)],[])
        self.init_action_set = []
        count = 0
        for i in range(len(self.deck)):
            for j in range(i+1,len(self.deck)):
                self.init_action_set.append( (count,(self.deck[i],self.deck[j])) )
                count += 1
        
    def get_action_set(self,info_set):
        stage = info_set[0]
        if stage == 0:
            return self.init_action_set
        elif stage == 3:
            card1,card2 = info_set[1]
            curr_deck = self.deck[:]
            curr_deck.remove(card1)
            curr_deck.remove(card2)
            return [(idx,item) for idx,item in enumerate(curr_deck)]
        else:
            return []
    
class Player1:
    def __init__(self,A_round1,A_round2):
        self.A_round1 = A_round1
        self.A_round2 = A_round2
        
    def get_action_set(self,info_set):
        stage = info_set[0]
        if stage == 1:
            return self.A_round1
        elif stage == 4:
            return self.A_round2
        else:
            return []
    
class Player2:
    def __init__(self,A_round1,A_round2):
        self.A_round1 = A_round1
        self.A_round2 = A_round2
        
    def get_action_set(self,info_set):
        stage = info_set[0]
        if stage == 2:
            return self.A_round1
        elif stage == 5:
            return self.A_round2
        else:
            return []

class LeducHistory:
    ## player idx: 0: chance, 1: p1, 2: p2

    def __init__(self):
        ## 初始化
        self.curr_player_idx = self.stage = 0
        self.action_history = tuple([])
        
    def copy(self):
        h = LeducHistory()
        h.curr_player_idx = self.curr_player_idx
        h.stage = self.stage
        h.action_history = self.action_history
        return h
    
    def is_terminated(self):
        return self.stage == 6
    
    def evaluation(self,p=1):
        assert self.is_terminated
        (card1,card2),bet11,bet12,public_card,bet21,bet22 = self.action_history
#         assert (not ((card1 == public_card) and (card2 == public_card)))
        bet1,bet2 = max(bet11,bet12),max(bet21,bet22)
        bet = bet1 + bet2 + 1
        if card1 == public_card:
            value = bet
        elif card2 == public_card:
            value = -bet
        else:
            if card1 > card2:
                value = bet
            elif card2 > card1:
                value = -bet
            else:
                value = 0
        if p == 1:
            return value
        elif p == 2:
            return -value
        else:
            return 0
        
    def get_player(self):
        return self.curr_player_idx
    
    def transition(self,a):
        if self.stage % 3 == 0:
            self.action_history = tuple(list(self.action_history) + [a[1]])
        else:
            self.action_history = tuple(list(self.action_history) + [a])
        self.stage = self.stage + 1
        self.curr_player_idx = (self.curr_player_idx + 1) % 3
    
    def get_infoset(self):
        if self.stage == 0:
            return (self.stage,)
        if self.stage == 1:
            (card1,card2), = self.action_history
            return (self.stage,card1)
        elif self.stage == 2:
            (card1,card2),bet11 = self.action_history
            return (self.stage,card2,bet11)
        elif self.stage == 3:
            (card1,card2),bet11,bet12 = self.action_history
            return (self.stage,(card1,card2),bet11,bet12)
        elif self.stage == 4:
            (card1,card2),bet11,bet12,public_card = self.action_history
            return (self.stage,card1,bet11,bet12,public_card)
        elif self.stage == 5:
            (card1,card2),bet11,bet12,public_card,bet21 = self.action_history
            return (self.stage,card2,bet11,bet12,public_card,bet21)
        else:
            return None