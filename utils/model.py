import torch
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for stock modelling embeddings.
    """

    def __init__(self, device="cpu"):
        super(BaseModel, self).__init__()
        self.embeddings = None
        self.device = torch.device(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, device="cpu"):
        # Load the state dict from the file
        state_dict = torch.load(path, map_location=torch.device(device))

        # Infer n_time_series and embedding_dim from the embeddings layer
        # Assuming the name of the embeddings layer in the state_dict is 'embeddings.weight'
        n_time_series, embedding_dim = state_dict["embeddings.weight"].shape

        # Create an instance of the cls with the inferred dimensions
        model = cls(n_time_series, embedding_dim)
        model.to(torch.device(device))

        # Load the state dict into the model
        model.load_state_dict(state_dict)

        return model

    def to_device(self, device):
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, *input):
        raise NotImplementedError("Forward method not implemented.")

    def calculate_loss(self, output, target):
        raise NotImplementedError("Loss calculation method not implemented.")

    def load_embeddings_from_numpy(self, embed_array):
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(embed_array), freeze=False
        )
        print(f"Number of Time Series: {self.embeddings.weight.shape[0]}")
        print(f"Embedding Dimension: {self.embeddings.weight.shape[1]}")

    def load_embeddings_from_csv(self, fname):
        self.load_embeddings_from_numpy(np.genfromtxt(fname=fname, delimiter=","))

    def save_embeddings_to_csv(self, fname):
        if fname.split(".")[1] != "csv":
            raise ValueError("You must include .csv in your file name")
        np.savetxt(fname, self.embeddings.weight.detach().numpy(), delimiter=",")

    def calculate_loss(self, output, target):
        # Implement a common method for calculating loss if applicable
        raise NotImplementedError("Loss calculation method not implemented.")


class NSE2Vec(BaseModel):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling.
    """

    def __init__(self, n_time_series: int, embedding_dim: int):
        super(NSE2Vec, self).__init__()
        self.embeddings = nn.Embedding(n_time_series, embedding_dim)

    def forward(self, inputs):
        # -- This extracts the relevant rows of the embedding matrix
        # - Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
        context_embeddings = self.embeddings(inputs)  # (batch_size, embed_dim)

        # -- Compute the hidden layer by a simple mean
        hidden = context_embeddings.mean(axis=1)  # (n_time_series, embed_dim)
        # -- Compute dot product of hidden with embeddings
        out = torch.einsum("nd,bd->bn", self.embeddings.weight, hidden)

        # -- Return the log softmax since we use NLLLoss loss function
        return nn.functional.log_softmax(out, dim=1)  # (batch_size, n_time_series)