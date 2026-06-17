import torch
import torch.nn as nn
from dingo.core.nn.nsf import create_nsf_wrapped
from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet
from torch.distributions import constraints
import math
from dingo.core.nn.enets import DenseResidualNet
from dingo.core.utils import torchutils
import copy

class CopulaNSFFlowWrapper(nn.Module):
    """
    similar to flowrapper, but for multiple flows
    """
    # TODO:: extend to N signals
    def __init__(self, posterior_kwargs, embedding_kwargs, initial_weights=None):
        super().__init__()
        
        self.theta_dim    = posterior_kwargs["input_dim"]
        self.context_dim  = posterior_kwargs["context_dim"]
        self.block_dims   = posterior_kwargs.get("block_dims", [2, 2])
        self.flow_1_args  = posterior_kwargs["flow_1"]
        self.flow_2_args  = posterior_kwargs["flow_2"]
        self.CopulaKwargs = posterior_kwargs["CopulaKwargs"]
        
        totaldim          = sum(self.block_dims)
        embedding_kwargs  = copy.deepcopy(embedding_kwargs)

        if initial_weights is not None:
            embedding_kwargs["V_rb_list"] = initial_weights["V_rb_list"]
        elif "V_rb_list" not in embedding_kwargs:
            embedding_kwargs["V_rb_list"] = None
            
        self.embedding_net = create_enet_with_projection_layer_and_dense_resnet(**embedding_kwargs)
        self.flow_1 = create_nsf_wrapped(
            input_dim=self.block_dims[0],
            context_dim=self.context_dim,
            **self.flow_1_args,
        )
        self.flow_2 = create_nsf_wrapped(
            input_dim=self.block_dims[1],
            context_dim=self.context_dim,
            ** self.flow_2_args,
        )
        p=self.CopulaKwargs.get("p", 1)
        self.vector_copula = VectorCopulaModel(self.flow_1,self.flow_2,p,totaldim,False,False,self.context_dim,self.CopulaKwargs)
    
    def log_prob(self, y, *x):
        if len(x) > 0:
            if self.embedding_net is not None:
                x = self.embedding_net(*x)
            return self.vector_copula.log_prob(y, x)
        else:
            return self.vector_copula.log_prob(y)
    
    def sample(self, *x, num_samples=1):
        if len(x) > 0:
            if self.embedding_net is not None:
                x = self.embedding_net(*x)
            return self.vector_copula.sample(x, num_samples=num_samples)
        else:
            return self.vector_copula.sample(num_samples)
    
    def sample_and_log_prob(self, *x, num_samples=1):
        if len(x) > 0:
            if self.embedding_net is not None:
                x = self.embedding_net(*x)
            return self.vector_copula.sample_and_log_prob(x, num_samples=num_samples)
        else:
            return self.vector_copula.sample_and_log_prob(num_samples)
    
    def forward(self, y, *x):
        if len(x) > 0:
            return self.log_prob(y, *x)
        else:
            return self.log_prob(y)
        
#TODO:: new version -> after test?
class VectorCopulaModel(nn.Module):
    arg_constraints = {}  # fill if you have constrained params
    support         = constraints.real_vector
    has_rsample     = True  # set True if you implement rsample()
    def __init__(self,flow_1,flow_2,p,d,isIndependentCopula,useIdentityTransform,context_dim,CopulaKwargs): 
        super().__init__()
        self.flow_1 = flow_1
        self.flow_2 = flow_2
        
        #size of B
        self.d      = d  
        self.p      = p
        
        self.marginal_dim = d // 2

        #Debug switches
        self.isIndependentCopula  = isIndependentCopula
        self.useIdentityTransform = useIdentityTransform
        self.context_dim = context_dim

        self.copulaNet = CopulaParamNet(self.context_dim,d,**CopulaKwargs)
        
    def log_prob(self, value: torch.Tensor, context) -> torch.Tensor:
        total,_,_,_,_,_ = self.logProbVectorCopula(value,context)
        return total
    
    def rsample(self,context, num_samples):
        if len(num_samples) == 0:
            return self._sampleVectorCopulaModel(
                context,1, 
            ).squeeze(0)

        N = math.prod(num_samples)
        
        return self._sampleVectorCopulaModel(
            context,N 
        )
    
    def sample(self, context, num_samples):
        with torch.no_grad():
            return self.rsample(context,num_samples)
        
    #TODO:: is context batched? -> can cause issues to be checked later
    def _buildOmega(self,context):
        B, zeta = self.copulaNet(context)

        eye = torch.eye(self.d, device=B.device, dtype=B.dtype)
        OmegaBar = zeta * eye + B @ B.T

        if not torch.isfinite(OmegaBar).all():
            raise RuntimeError("OmegaBar contains NaN/Inf")

        dimList = [self.marginal_dim, self.marginal_dim]
        Bd = Blockdiag(OmegaBar, dimList)

        if not torch.isfinite(Bd).all():
            raise RuntimeError("Blockdiag(OmegaBar) contains NaN/Inf")

        Bd = Bd + 1e-6 * eye
        L = torch.linalg.cholesky(Bd)

        A = torch.linalg.solve_triangular(L, eye, upper=False)
        Omega = A @ OmegaBar @ A.T

        if not torch.isfinite(Omega).all():
            raise RuntimeError("Omega contains NaN/Inf")

        return Omega

    def logProbVectorCopula(self,theta, context):
        if theta.ndim == 2:
            # batched
            zeros = theta.new_zeros(theta.shape[0])
        else:
            # single event
            zeros = theta.new_tensor(0.0)
        theta0 = theta[:,: self.marginal_dim]
        theta1 = theta[:,self.marginal_dim :]
        #get log prob of the marginal flows using made products
        if self.useIdentityTransform:
            logp_marg_1 = zeros
            logp_marg_2 = zeros
            #generate input for log prob copula

            Q_1 = theta[:,: self.marginal_dim]
            Q_2 = theta[:,self.marginal_dim :]
        else:
            logp_marg_1 = self.flow_1.log_prob(theta0, context)
            logp_marg_2 = self.flow_2.log_prob(theta1, context)
            #generate input for log prob copula
            Q_1,_ = self.flow_1.flow._transform.inverse(theta0, context=context)
            Q_2,_ = self.flow_2.flow._transform.inverse(theta1, context=context)
            
        if len(theta.shape) == 2:
            dim = 1
        else: #len = 1
            dim = 0
        Q = torch.concat([Q_1,Q_2],dim = dim)
        
        # TODO:: fix logprobCopulas conditional
        logDensity,logDetTerm,logCopulaTerm  = self._logProbCopula(Q,context)
        total = logp_marg_1 + logp_marg_2 + logDensity
        return total,logp_marg_1,logp_marg_2,logDensity,logDetTerm,logCopulaTerm
    
    def sample_and_log_prob(self,N, context):
        samples = self.sample(context,num_samples=N)
        log_q   = self.log_prob(samples, context)
        return samples,log_q
    
    def _sampleGaussianBase(self,N,context):
        if self.isIndependentCopula:
            Omega = torch.eye(self.d)
        else:
            Omega = self._buildOmega(context)
        dist = torch.distributions.MultivariateNormal(loc=torch.tensor(torch.zeros(self.d))
                                                    , covariance_matrix=Omega)
        sample = dist.rsample((N,))
        return sample

    def _sampleVectorCopula(self,N,context):
        Z  = self._sampleGaussianBase(N,context)  # replace with multivariate distr
        return Z[:,:self.marginal_dim],Z[:,self.marginal_dim:]

    def _sampleVectorCopulaModel(self,context,N):
        dist_1 = self.flow_1()
        dist_2 = self.flow_2()
        if self.isIndependentCopula:
            sample_1 = dist_1.rsample((N,))
            sample_2 = dist_2.rsample((N,))
        else:
            Z_1,Z_2 = self._sampleVectorCopula(N,context)
            if self.useIdentityTransform:
                sample_1 = torch.randn(N, self.marginal_dim)
                sample_2 = torch.randn(N, self.marginal_dim)
            else:
                sample_1 = dist_1.transform(Z_1)  #numerical shortcut can be removed so no \phi(\phi^-1)) be used because they are independent
                sample_2 = dist_2.transform(Z_2)
        return torch.cat([sample_1,sample_2], dim=1)

    def _logProbCopula(self,Q,context):
        if Q.ndim == 2:
            # batched
            zeros = Q.new_zeros(Q.shape[0])
        else:
            # single event
            zeros = Q.new_tensor(0.0)
        
        if self.isIndependentCopula:
            return zeros,zeros,zeros
        Omega           = self._buildOmega(context)
        I               = torch.eye(self.d)
        L = torch.linalg.cholesky(Omega)
        OmegaInv = torch.cholesky_solve(I, L)
        logabsdet = 2 * torch.log(torch.diagonal(L)).sum()
        PhiInv          = Q
        if len(Q.shape) == 2:
            einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (OmegaInv- I), PhiInv)  #sum in order to deal with quadratic form dimensions
        else: #len = 1 so one event
            einsum = PhiInv @ (OmegaInv- I) @ PhiInv
        logDetTerm      = (-1/2)*logabsdet
        logCopulaTerm   = (-1/2)*einsum
        logDensity      = logDetTerm + logCopulaTerm #for exact expression, constant is needed-> not used
        return logDensity,logDetTerm,logCopulaTerm

class CopulaParamNet(nn.Module):
    def __init__(
            self,
            context_dim: int,
            d: int,  #sum of the blocs
            **CopulaKwargs
        ):
        super().__init__()
        self.d           = d
        self.p           = CopulaKwargs['p']
        output_dim       = self.d*self.p + 1  #B iis a dxp matrix +1 for zeta
        self.hidden_dims = CopulaKwargs['hidden_dims']
        activation_fn    = torchutils.get_activation_function_from_string(CopulaKwargs['activation'])
        
        self.net = DenseResidualNet(
            input_dim        = context_dim,
            output_dim       = output_dim,
            hidden_dims      = CopulaKwargs['hidden_dims'],
            activation       = activation_fn,
            dropout          = CopulaKwargs['dropout'],
            batch_norm       = CopulaKwargs['batch_norm'], #good to be on since copulas are affine invariant -> normalization removes these transforms
            context_features = None,
        )
        if self.p >= self.d:
            raise ValueError(f"{self.p} is not strictly smaller than {self.d}")
        
    def forward(self, context):
        raw   = self.net(context)
        B_raw = raw[:,:self.d*self.p]
        zeta  = torch.nn.functional.softplus(raw[:,self.d*self.p]) + 1e-6 #enforce numerical stability
        B     = B_raw.reshape(-1, self.d, self.p)
        return B, zeta

def Blockdiag(B,dimList):
    Bdiag = B.clone()
    prevDim = 0
    for dim in dimList:
        ind = dim + prevDim
        Bdiag[:ind,ind:] = 0
        Bdiag[ind:,:ind] = 0   
        prevDim += dim  
        
    # print(f"is PSD: {is_psd(Bdiag)}")
    return Bdiag
