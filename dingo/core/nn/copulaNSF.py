import torch
import torch.nn as nn
from dingo.core.nn.nsf import create_nsf_wrapped
from dingo.core.nn.enets import create_enet_with_projection_layer_and_dense_resnet
from torch.distributions import constraints
import math
from dingo.core.nn.enets import DenseResidualNet
from dingo.core.utils import torchutils
import copy
from vector_copula_vi import AmortizedVectorCopulaFlow,DingoVectorCopulaFlow

class CopulaNSFFlowWrapper(nn.Module):
    """
    similar to flowrapper, but for multiple flows
    """
    # TODO:: extend to N signals
    def __init__(self, posterior_kwargs, embedding_kwargs, initial_weights=None):
        super().__init__()
        
        self.conditional = posterior_kwargs.get("conditional", True)
        if "context_dim" not in posterior_kwargs:
            self.context_dim = None
        else:
            self.context_dim  = posterior_kwargs["context_dim"]
            
        self.block_dims   = posterior_kwargs.get("block_dims", [2, 2])
        self.D            = sum(self.block_dims)

        self.flows_args   = posterior_kwargs["flows"]
        self.CopulaKwargs = posterior_kwargs["CopulaKwargs"]
        
        embedding_kwargs  = copy.deepcopy(embedding_kwargs)
        if not self.conditional:
            self.embedding_net = None
        else:
            if initial_weights is not None:
                embedding_kwargs["V_rb_list"] = initial_weights["V_rb_list"]
            elif "V_rb_list" not in embedding_kwargs:
                embedding_kwargs["V_rb_list"] = None
            
            self.embedding_net = create_enet_with_projection_layer_and_dense_resnet(**embedding_kwargs)
        assert len(posterior_kwargs["flows"]) == len(self.block_dims), (
            f"Expected one flow per block dimension, but got "
            f"{len(posterior_kwargs['flows'])} nbr of flows and {len(self.block_dims)} block dimensions."
        )
        
        flows = nn.ModuleList()

        for i, (flow_name, flow_args) in enumerate(
            posterior_kwargs["flows"].items()
        ):
            flow = create_nsf_wrapped(
                input_dim=self.block_dims[i],
                context_dim=self.context_dim,
                **flow_args,
            )

            flows.append(flow)
        
        if self.conditional:
            self.copulaNet = CopulaParamNet(
                self.context_dim,
                self.D,
                **self.CopulaKwargs,
            )

            self.vector_copula = AmortizedVectorCopulaFlow(
                flows,
                self.copulaNet,
                marginal_backend="dingo",
                is_independent = self.CopulaKwargs.get("is_independent", False)
            )

        else:
            self.flows         = flows
            self.copulaNet     = None
            self.vector_copula = None
            P = self.CopulaKwargs["P"]
            self.is_independent = self.CopulaKwargs.get("is_independent", False)

            if P >= self.D:
                raise ValueError(
                    f"Copula rank P={P} must be smaller than D={self.D}."
                )

            self.B = nn.Parameter(
                0.01 * torch.randn(1, self.D, P)
            )

            self.z = nn.Parameter(
                torch.zeros(1)
            )
    def distribution(self, context=None):
        if self.conditional:
            if context is None:
                raise ValueError(
                    "Conditional copula requires context."
                )

            # This calls CopulaParamNet(context) internally.
            return self.vector_copula#.distribution(context)

        if context is not None:
            raise ValueError(
                "Unconditional copula must not receive context."
            )

        # Non-amortized path: B and z are direct global parameters.
        return DingoVectorCopulaFlow(
            flows=self.flows,
            B=self.B,
            z=self.z,
            context=None,
            is_independent = self.is_independent
        )
    def log_prob(self, y, *x):
        if len(x) > 0:
            if self.embedding_net is not None:
                x = self.embedding_net(*x)
            return self.distribution(context=x).log_prob(y, x)
        else:
            return self.distribution().log_prob(y)
    
    def sample(self, *x, num_samples=1):
        if len(x) > 0:
            if self.embedding_net is not None:
                x = self.embedding_net(*x)
            return self.distribution(context=x).sample(sample_shape=num_samples, context = x)
        else:
            return self.distribution().sample(sample_shape = num_samples)
    
    def sample_and_log_prob(self, *x, num_samples=1):
        if len(x) > 0:
            if self.embedding_net is not None:
                x = self.embedding_net(*x)
            return self.distribution(context=x).sample_and_log_prob(context = x, N=num_samples)
        else:
            return self.distribution().sample_and_log_prob(N=num_samples)
    
    def forward(self, y, *x):
        if len(x) > 0:
            return self.log_prob(y, *x)
        else:
            return self.log_prob(y)
    
class CopulaParamNet(nn.Module):
    def __init__(
            self,
            context_dim: int,
            D: int,  #sum of the blocs
            **CopulaKwargs
        ):
        super().__init__()
        self.D           = D
        self.P           = CopulaKwargs['P']
        output_dim       = self.D*self.P + 1  #B iis a dxp matrix +1 for zeta
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
        if self.P >= self.D:
            raise ValueError(f"{self.P} is not strictly smaller than {self.D}")
    
    def forward(self, context):
        raw = self.net(context)

        B_raw = raw[:, :self.D * self.P]

        z = raw[:, self.D * self.P]

        B = B_raw.reshape(
            -1,
            self.D,
            self.P,
        )

        return B, z

