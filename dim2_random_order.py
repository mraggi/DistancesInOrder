from differential_evolution import optimize
from timer import Timer
import torch
import torch.nn.functional as F
from time import sleep
from itertools import permutations
import random

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

class PermCost():
    def __init__(self, P, num_points_left_side, default_points):
        self.P = P
        self.n = num_points_left_side
        self.T = default_points
        
    def __call__(self, A):
        T,P,n = self.T,self.P,self.n
        
        W = add_points(T,A)
        B,C = W[:,:n], W[:,n:]
        D = pairwise_distances(B,C)
        bs = D.shape[0]
        D = D.view(bs,-1)
        S,_ = D.sort(dim=1)
        
        #raise "BLA"
        P.to(D.device)
        #print(f"\nP.shape = {P.shape}")
        ##print(f"D.shape = {D.shape}")
        #print(f"B.shape = {B.shape}")
        #print(f"C.shape = {C.shape}")
        #print(f"D[:,P].shape = {D[:,P].shape}")
        #print(f"S.shape = {S.shape}")
        return torch.sum(torch.abs(D[:,P] - S), dim=1) + too_close_loss(D) + distances_too_similar(D)
    
bad_ones = []
n,m = 3,4
def_pts = [[-1,0],[1,0]]
device = 'cuda'
X = [list(range(n*m)) for _ in range(10000)]

for i,x in enumerate(X):
    random.shuffle(X[i])
#X = [[0] + list(p) for p in permutations(range(1,n*m))]

#X = bad_ones

print(f"Trying {len(X)} permutations")

def do_perm(p,num_tries=1):
    best_value, best_x = 99999999, 0
    
    print(f"\n\nAttempting permutation {p}")
    
    for _ in range(num_tries):
        perm = torch.tensor(p,device=device)
        print(f"\tAttempt {_}: ",end="")
        initial_pop = torch.randn((512,n+m-len(def_pts),2),device=device).double()*2
    
        value, x = optimize(PermCost(perm,n,def_pts), 
                            num_populations=32, initial_pop=initial_pop,
                            mut=(0.3,0.9),
                            crossp=(0.3,0.9),
                            epochs=12000, 
                            shuffles=3,
                            proj_to_domain=lambda x: torch.clamp(x,-30,30), #+ torch.randn_like(x)*0.00001, 
                            use_cuda=False,
                            break_at_cost=0
                        )
        if value == 0:
            print("SUCCESS!\n")
            return value, x
        else:
            print("FAIL!\n")
        if value < best_value:
            best_value, best_x = value, x
    return best_value, best_x

for i,p in enumerate(X):
    
    value,x = do_perm(p,20)
    try:
        sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupting!! Bad ones so far: ", bad_ones)
        raise
    
    if value != 0:
        print(f"\n\n\n\n\n******************************************BAD permutation: \n{p}\nBest value: {value}\n\n")
        W = add_points(def_pts,x[None])
        B,C = W[:,:n], W[:,n:]
        print(f"{100.*i/len(X):.4f}%: Perm {p} CANNOT be realized. Closest I found: \n{B}¸and \n{C}\n")
        
        bad_ones += [p]
        print(f"{i+1} attempted, and bad ones contains now {len(bad_ones)}")
        
    else:
        W = add_points(def_pts,x[None])
        B,C = W[:,:n], W[:,n:]
        print(f"\n{100.*i/len(X):.3f}%: Perm {p} can be realized with \n{B}¸and \n{C}\n Distance matrix:\n{pairwise_distances(B,C)}")
            
        
print("DONE! Bad ones for now: ", bad_ones)
