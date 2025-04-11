import torch
from fastprogress import progress_bar
from helpers import *

def individual2population(f):
    return lambda P : torch.stack([f(p) for p in P])

class DifferentialEvolver:
    def __init__(self, f, 
                       initial_pop = None, # In next version, I'll remove the option of giving pop_size, dims and num_populations instead of initial_population
                       pop_size=50, dim = (1,), # ignored if initial_pop is given
                       num_populations=1, # If initial_pop is given, then num_populations must divide initial_pop.shape[0]
                       proj_to_domain = lambda x : x, 
                       f_for_individuals = False, proj_for_individuals = None,
                       maximize = False,
                       use_cuda = False,
                       prob_choosing_method = 'automatic', # either 'randint', 'multinomial' or 'automatic'
                       chromosome_replacement_dimension = None, # None means that every single number could be replaced independently from others.
                                                               # 0 means the whole individual is either replaced or not (stupid!)
                                                               # 1 means every component of the individual is either replaced or not, etc.
                ):
        
        if isinstance(dim,int): dim = (dim,)
        
        if initial_pop is None: P = torch.randn(pop_size*num_populations, *dim)
        else: P = initial_pop
        
        self.pop_size, *self.dim = P.shape
        self.num_populations = num_populations
        assert(self.pop_size%self.num_populations == 0)
        block_size = self.pop_size//self.num_populations
        
        if proj_for_individuals is None: proj_for_individuals = f_for_individuals

        if f_for_individuals: f = individual2population(f)
        if proj_for_individuals: proj_to_domain = individual2population(proj_to_domain)
        
        if use_cuda: P = P.cuda()
        
        P = proj_to_domain(P)

        self.use_randint = (prob_choosing_method in ['randint', 'random', 'rand_int'])
        
        if prob_choosing_method in ['automatic', 'auto', None]: self.use_randint = (block_size >= 100)
        
        if self.use_randint:
            n = self.pop_size
            s = self.num_populations
            b = n//s
            if s == 1: 
                self._rand_indices = lambda : torch.randint(0,n,(3,n),device=P.device)
            else: 
                S = torch.arange(s,device=P.device).repeat_interleave(b)[None].contiguous()
                self._rand_indices = lambda : S + torch.randint(0,b,(3,n),device=P.device)
        else:
            self.idx_prob = get_block_eye(block_size,self.num_populations).to(P)
        
        self.f = f if not maximize else (lambda x: -f(x)) 
        self.cost = self.f(P).squeeze()
        self.P = P
        self.proj_to_domain = proj_to_domain
        self.maximize = maximize
        
        self._dims_1 = tuple([self.pop_size] + [1 for _ in self.dim])
        
        crp = chromosome_replacement_dimension
        if crp is None: crp = len(self.dim)
        self._crossp_dims = tuple([self.pop_size] + [d for d in self.dim[:crp]] + [1 for _ in self.dim[crp:]])
        
    def _cross_pollination(self, crossp):
        return (torch.rand(self._crossp_dims, device=self.P.device) < crossp).to(self.P)
    
    def shuffle(self):
        I = torch.randperm(self.P.shape[0], device=self.P.device)
        self.P = self.P[I]
        self.cost = self.cost[I]
     
    def step(self, mut=0.8, crossp=0.7):
        A,B,C = self._get_ABC()
        
        mutants = A + mut*(B - C)
        
        T = self._cross_pollination(crossp)
        
        candidates = self.proj_to_domain(T*mutants + (1-T)*self.P)
        f_candidates = self.f(candidates).squeeze()
        
        should_replace = (f_candidates <= self.cost)
        
        self.cost = torch.where(should_replace,f_candidates,self.cost)
        
        # adjust dimensions for broadcasting
        S = should_replace.to(self.P).view(*self._dims_1) 
        
        self.P = S*candidates + (1-S)*self.P
            
    def best(self):
        best_cost, best_index = torch.min(self.cost, dim=0)
        if self.maximize:
            best_cost *= -1
            
        return best_cost.item(), self.P[best_index]
        
    def _get_ABC(self):
        I = self._rand_indices() if self.use_randint else torch.multinomial(self.idx_prob,3).T
        return self.P[I]
    
    
def optimize(f, initial_pop = None, 
                pop_size=20, dim = (1,), 
                num_populations=1, shuffles = 0,
                mut=0.8, crossp=0.7,  
                epochs=1000,
                proj_to_domain = lambda x : x, 
                f_for_individuals = False, proj_for_individuals = None, 
                maximize = False,
                use_cuda = False,
                prob_choosing_method = 'automatic',
                chromosome_replacement_dimension = 1,
                break_at_cost = None,
                mb = None
            ):
    
    if num_populations == 1: shuffles = 0 # no point in shuffling otherwise!!
        
    D = DifferentialEvolver(f=f, 
                            initial_pop=initial_pop,
                            pop_size=pop_size, dim = dim, 
                            num_populations=num_populations,
                            proj_to_domain = proj_to_domain, 
                            f_for_individuals = f_for_individuals, 
                            proj_for_individuals = proj_for_individuals,
                            maximize=maximize,
                            use_cuda=use_cuda,
                            prob_choosing_method=prob_choosing_method,
                            chromosome_replacement_dimension = chromosome_replacement_dimension
                           )
    if isinstance(epochs, int): epochs = range(epochs)
    mut, crossp = tofunc(mut), tofunc(crossp)

    pbar = progress_bar(epochs, parent=mb)
        
    
    test_each = 1
    
    try:
        remaining_before_test = test_each+1
        
        i = 0
        shuffles_so_far = 0
        
        for _ in pbar:
            remaining_before_test -= 1
            D.step(mut=mut(), crossp=crossp())
            
            i += 1
            progress = i/pbar.total
            
            if progress > (shuffles_so_far+1)/(shuffles+1):
                shuffles_so_far += 1
                D.shuffle()
            
            if remaining_before_test == 0:
                remaining_before_test = test_each
                best_cost, _ = D.best()
                pbar.comment = f"| best cost = {best_cost:.4f}"
                if break_at_cost is not None and best_cost == break_at_cost:
                    break
            
    except KeyboardInterrupt:
        print("Interrupting! Returning best found so far")
    
    return D.best()
