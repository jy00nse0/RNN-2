import torch
from model.seq2seq.embeddings import embeddings_factory
from model.seq2seq.encoder import encoder_factory
from model.seq2seq.decoder import decoder_factory
from model.seq2seq.model import Seq2SeqTrain, Seq2SeqPredict
from collections import OrderedDict


def train_model_factory(args, src_metadata, tgt_metadata):
    """
    Build Seq2Seq model with SRC metadata for Encoder and TGT metadata for Decoder.
    Output vocabulary size is TGT vocab size.
    """
    # Separate embeddings per module to match vocabularies
    encoder = encoder_factory(args, src_metadata, embed=None)
    decoder = decoder_factory(args, tgt_metadata, embed=None)
    
    # Pass tgt_pad_idx from metadata (NOT hardcoded)
    return Seq2SeqTrain(
        encoder, 
        decoder, 
        tgt_metadata.vocab_size, 
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        tgt_pad_idx=tgt_metadata.padding_idx
    )


def predict_model_factory(args, src_metadata, tgt_metadata, model_path, src_field, tgt_field):
    train_model = train_model_factory(args, src_metadata, tgt_metadata)
    train_model.load_state_dict(get_state_dict(args, model_path))
    return Seq2SeqPredict(train_model.encoder, train_model.decoder, src_field, tgt_field)


def get_state_dict(args, model_path):
    # load state dict and map it to current storage (CPU or GPU)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    # if model was trained with DataParallel (on multiple GPUs) remove "module." at the beginning of every key in state
    # dict (so we can load model on 1 GPU or on CPU for inference)
    if args.cuda and args.multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:]  # remove "module."
            new_state_dict[key] = v
        return new_state_dict

    return state_dict
