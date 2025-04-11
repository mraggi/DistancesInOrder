from differential_evolution import optimize
from timer import Timer
import torch
import torch.nn.functional as F
from time import sleep
from itertools import permutations
import random
from fastprogress import progress_bar, master_bar

def pairwise_distances(A,B):
    # Both A and B should be (bs,n,d) where n is some number of points and d is dimension (e.g. 2 for R^2)
    w = (A[:,:,None] - B[:,None])
    d2 = (w*w).sum(dim = -1)
    return torch.sqrt(d2)

def inverse_perm(P):
    Q = [0]*len(P)
    for i,p in enumerate(P):
        Q[p] = i
    return Q

def add_points(T, A):
    T = torch.tensor(T,device=A.device,dtype=A.dtype)
    bs = A.shape[0]
    W = torch.cat([T.repeat(bs,1,1),A],dim=1)
    return W

def too_close_loss(D):
    β = (D < 0.01).double()
    return β.sum(dim=1)

def distances_too_similar(D):
    U = pairwise_distances(D[...,None],D[...,None])
    u = U.shape[1]

    i = torch.arange(u,device=U.device)
    U[:,i,i] = 1
    U = U.view(U.shape[0],-1)
    β = (U < 0.01).double()
    return β.sum(dim=1)

def left_perm(a, m):
    A = list(a)
    n = len(A)
    return [a+t for t in range(0,n*m,n) for a in A]

def right_perm(a, n):
    A = list(a)
    m = len(A)
    return [a+t for a in A for t in range(0,n*m,m) ]

class Dummydict():
    def __init__(self, X):
        self.Dict = {tuple(x):0 for x in X}
        self.num_burned = 0
    
    def burn_Q(self,Q):
        Q = Q.cpu()
        for a in permutations(range(n)):
            for b in permutations(range(m)):
                Q = Q[:,left_perm(a,m)]
                Q = Q[:,right_perm(b,n)]
                for q in Q:
                    p = tuple([t.item() for t in q])
                    self.burn(p)
    
    def burn(self,p):
        if p in self.Dict.keys():
            self.Dict[p] += 1
            if self.Dict[p] == 1:
                self.num_burned += 1
                if self.num_burned%10000 == 0:
                    print(f"Total Burned: {self.num_burned/len(self.Dict)*100.:.5f}%")
                    
    
    def has(self, p):
        return self.Dict[tuple(p)] > 0
    
    def missing(self):
        return [k for k,v in self.Dict.items() if v == 0]
    
class PermCost():
    def __init__(self, P, num_points_left_side : int, default_points, dummy : Dummydict):
        self.P = P
        self.n = num_points_left_side
        self.T = default_points
        self.dummy = dummy
        
    def __call__(self, A):
        T,P,n = self.T,self.P,self.n
        
        W = add_points(T,A)
        B,C = W[:,:n], W[:,n:]
        D = pairwise_distances(B,C)
        bs = D.shape[0]
        D = D.view(bs,-1)
        S,Q = D.sort(dim=1)
        self.dummy.burn_Q(Q)
        
        P.to(D.device)
        
        return torch.sum(torch.abs(D[:,P] - S), dim=1) + too_close_loss(D) + distances_too_similar(D)

n,m = 3,4

def do_perm(p, dummy, num_tries=1, mb=None):
    best_value, best_x = 99999999, 0
    
    #print(f"\n\nAttempting permutation {p}")
    device = 'cuda'
    for _ in range(num_tries):
        perm = torch.tensor(p,device=device)
        #print(f"\tAttempt {_}: ",end="")
        initial_pop = torch.randn((2048,n+m-len(def_pts),2),device=device).double()*random.random()*10
    
        value, x = optimize(PermCost(perm,n,def_pts, dummy), 
                            num_populations=128, initial_pop=initial_pop,
                            mut=(0.3,0.9),
                            crossp=(0.3,0.9),
                            epochs=1000, 
                            shuffles=1,
                            proj_to_domain=lambda x: torch.clamp(x,-30,30), #+ torch.randn_like(x)*0.00001, 
                            use_cuda=True,
                            break_at_cost=0,
                            mb=mb
                        )
        if value == 0:
            #print("SUCCESS!\n")
            return value, x
        else:
            pass
            #print("FAIL!\n")
        if value < best_value:
            best_value, best_x = value, x
    return best_value, best_x


X = []

for p in permutations(range(2,n*m)):
    p = list(p)
 
    X.append([0, 1] + p)
    X.append([0] + p[:m-1] + [1] + p[m-1:])
    X.append([0] + p + [1])
    
    
print(f"Will try {len(X)} permutations")
    
dummy = Dummydict(X)

def_pts = [[-1,0],[1,0]]
device = 'cuda'

bad_ones = []
mbar = master_bar(X)
for i,p in enumerate(mbar):
    if dummy.has(p): continue
    
    mbar.main_bar.comment = f"| Total Burned: {100.*dummy.num_burned/len(X):.3f}%"
    
    value,x = do_perm(p,dummy,1,mb=mbar)
    
    if value != 0:
        print(f"\n\n\n\n\n******************************************BAD permutation: \n{p}\nBest value: {value}\n\n")
        W = add_points(def_pts,x[None])
        B,C = W[:,:n], W[:,n:]
        print(f"{100.*i/len(X):.4f}%: Perm {p} CANNOT be realized. Closest I found: \n{B}¸and \n{C}\n")
        
        bad_ones += [p]
        print(f"{i+1} attempted, and bad ones contains now {len(bad_ones)}")
        try:
            sleep(0.25)
        except KeyboardInterrupt:
            print("Interrupting!! Bad ones so far: ", bad_ones)
            raise
    else:
        W = add_points(def_pts,x[None])
        B,C = W[:,:n], W[:,n:]
        #print(f"\n{100.*i/len(X):.3f}%: Perm {p} can be realized with \n{B}¸and \n{C}\n Distance matrix:\n{pairwise_distances(B,C)}")
            
        
print("DONE! Bad ones for now: ", bad_ones)