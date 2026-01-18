import torch.nn as nn
import torch


def embeddings_factory(args, metadata):
    if metadata.vectors is not None:
        embed = nn.Embedding(
            num_embeddings=metadata.vocab_size,
            embedding_dim=args.embedding_size,
            padding_idx=metadata.padding_idx,
            _weight=metadata.vectors
        )
    else:
        embed = nn.Embedding(
            num_embeddings=metadata.vocab_size,
            embedding_dim=args.embedding_size,
            padding_idx=None
        )
        with torch.no_grad():
            nn.init.uniform_(embed.weight, -0.1, 0.1)

    # padding row는 항상 보정
    with torch.no_grad():
        embed.weight[metadata.padding_idx].fill_(0)

    embed.weight.requires_grad = args.train_embeddings
    return embed

