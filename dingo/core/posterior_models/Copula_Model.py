from .base_model import BasePosteriorModel
from dingo.core.nn.copulaNSF import CopulaNSFFlowWrapper

# TODO: fix batching in copula's (now we have multiple events)
class CopulaNormalizingFlowModel(BasePosteriorModel):
    """
    DINGO-compatible wrapper around a conditional copula guide.

    This class is not the architecture itself. It is the DINGO-facing
    posterior model object used by prepare_training_new(...) and train_stages(...).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def initialize_network(self):
        model_kwargs = {
            k: v for k, v in self.model_kwargs.items()
            if k != "posterior_model_type"
        }
            
        posterior_kwargs = model_kwargs["posterior_kwargs"]
        embedding_kwargs = model_kwargs.get("embedding_kwargs", None)
        
        self.network = CopulaNSFFlowWrapper(
            posterior_kwargs=posterior_kwargs,
            embedding_kwargs=embedding_kwargs,
            initial_weights=self.initial_weights
        )
        
    def log_prob(self, theta, *context):
        """
        Parameters
        ----------
        theta:
            Tensor of shape [B, D].
            These are DINGO-standardized inference parameters.

        context:
            DINGO context tensors, usually strain/ASD embeddings and possibly
            additional GNPE proxies.

        Returns
        -------
        log_q:
            Tensor of shape [B].
        """
        return self.network.log_prob(theta, *context)

    def sample(self, *context, num_samples: int = 1):
        """
        Returns samples of shape [B, num_samples, D].
        """
        return self.network.sample(*context, num_samples=num_samples)

    def sample_and_log_prob(self, *context, num_samples: int = 1):
        """
        Returns:
            samples:  [B, num_samples, D]
            log_prob: [B, num_samples]
        """
        return self.network.sample_and_log_prob(*context, num_samples=num_samples)

    def loss(self, theta, *context):
        """
        Standard conditional density-estimation objective:
            - E_{theta, X ~ training data} log q_phi(theta | X)
        """
        return -self.log_prob(theta, *context).mean()