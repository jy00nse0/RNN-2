import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from util import Metadata
import html
import numpy as np


class Vocab:
    """Simple vocabulary class"""
    def __init__(self, tokens, specials=['<pad>', '<sos>', '<eos>','<unk>']):
        self.specials = specials
        self.itos = specials + sorted(list(set(tokens) - set(specials)))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_index = self.stoi['<unk>']
        
    def __len__(self):
        return len(self.itos)
    
    def encode(self, tokens):
        return [self.stoi. get(tok, self.unk_index) for tok in tokens]
    
    def decode(self, indices):
        return [self.itos[idx] for idx in indices]

def limit_vocab(tokens_iterator, max_size=50000):
    """
    Limit vocabulary to top max_size frequent words.
    """
    # Count all tokens
    counter = Counter(tokens_iterator)
    # Get top max_size tokens
    most_common = counter.most_common(max_size)
    # Return list of tokens
    return [token for token, count in most_common]


class BucketBatchSampler(Sampler):
    """
    Length-bucketed batch sampler for efficient training.
    
    Sorts dataset indices by example length (short to long), forms batches
    of size `batch_size`, and shuffles the order of batches per epoch using
    `set_epoch(epoch)`.
    
    Usage:
        Enable with args.bucket_batching=True in dataset_factory.
        Call sampler.set_epoch(epoch) at the start of each training epoch.
    
    Args:
        dataset_lengths: Tensor or list of sequence lengths for each example
        batch_size: Number of examples per batch
        drop_last: Whether to drop the last incomplete batch
    """
    def __init__(self, dataset_lengths, batch_size, drop_last=False):
        self.dataset_lengths = dataset_lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0
        
        # Sort indices by length (short to long)
        if isinstance(dataset_lengths, torch.Tensor):
            sorted_indices = torch.argsort(dataset_lengths).tolist()
        else:
            sorted_indices = sorted(range(len(dataset_lengths)), 
                                  key=lambda i: dataset_lengths[i])
        
        # Form batches from sorted indices
        self.batches = []
        for i in range(0, len(sorted_indices), batch_size):
            batch = sorted_indices[i:i + batch_size]
            if len(batch) == batch_size or not drop_last:
                self.batches.append(batch)
    
    def set_epoch(self, epoch):
        """Set epoch for shuffling batch order"""
        self.epoch = epoch
    
    def __iter__(self):
        # Shuffle batch order based on epoch (deterministic per epoch)
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.batches), generator=g).tolist()
        
        for idx in indices:
            yield self.batches[idx]
    
    def __len__(self):
        return len(self.batches)


class LazyTranslationDataset(Dataset):
    """
    Lazy loading dataset for translation tasks.
    Indexes file offsets on initialization and reads lines on-demand.
    Minimal memory footprint suitable for multiprocessing.
    """
    def __init__(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None, reverse_src=False, max_len=50):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.reverse_src = reverse_src
        self.max_len = max_len
        
        # 1. Build line offsets
        print(f"Indexing {src_file}...")
        self.src_offsets = self._build_offsets(src_file)
        print(f"Indexing {tgt_file}...")
        self.tgt_offsets = self._build_offsets(tgt_file)
        
        assert len(self.src_offsets) == len(self.tgt_offsets), \
            f"Line count mismatch: SRC {len(self.src_offsets)} vs TGT {len(self.tgt_offsets)}"
        
        # 2. Build or use provided vocabularies
        # To build vocab, we must scan the file once.
        if src_vocab is None:
            print("Building source vocabulary...")
            src_tokens = self._scan_for_vocab(src_file, reverse_src)
            # Limit to top 50k
            src_tokens_limited = limit_vocab(src_tokens, max_size=50000)
            self.src_vocab = Vocab(src_tokens_limited)
        else:
            self.src_vocab = src_vocab
            
        if tgt_vocab is None:
            print("Building target vocabulary...")
            tgt_tokens = self._scan_for_vocab(tgt_file, False)
            # Limit to top 50k
            tgt_tokens_limited = limit_vocab(tgt_tokens, max_size=50000)
            self.tgt_vocab = Vocab(tgt_tokens_limited)
        else:
            self.tgt_vocab = tgt_vocab
        
        # Cached lengths for bucket batching (computed lazily on first access)
        self._lengths_cache = None
    
    def compute_lengths(self):
        """
        Compute source sequence lengths for all examples efficiently.
        Uses file offsets to read only line lengths without loading full text.
        Returns tensor of lengths (including <eos> token).
        """
        if self._lengths_cache is not None:
            return self._lengths_cache
        
        print(f"Computing lengths for bucket batching...")
        lengths = []
        with open(self.src_file, 'r', encoding='utf-8') as f:
            for offset in self.src_offsets:
                f.seek(offset.item())
                line = f.readline().strip()
                # Count tokens (will add <eos>, so +1)
                token_count = len(line.split())
                # Apply max_len truncation
                token_count = min(token_count, self.max_len)
                # Add 1 for <eos> token
                lengths.append(token_count + 1)
        
        self._lengths_cache = torch.tensor(lengths, dtype=torch.int64)
        return self._lengths_cache
            
    def _build_offsets(self, path):
        offsets = [0]
        with open(path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                offsets.append(f.tell())
        # Remove last offset which points to EOF
        offsets.pop() 
        return torch.tensor(offsets, dtype=torch.int)
    
    def _scan_for_vocab(self, path, reverse):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                # Apply HTML unescape to handle entities
                tokens = [html.unescape(t) for t in tokens]
                yield from tokens

    def _read_line(self, path, offset):
        with open(path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return f.readline().strip().split()

    def __len__(self):
        return len(self.src_offsets)
    
    def __getitem__(self, idx):
        # 1. Read Source
        src_tokens = self._read_line(self.src_file, self.src_offsets[idx].item())
        if self.reverse_src:
            src_tokens = list(reversed(src_tokens))
        
        # Truncate
        if len(src_tokens) > self.max_len:
            src_tokens = src_tokens[:self.max_len]
            
        src = src_tokens + ['<eos>']
        
        # 2. Read Target
        tgt_tokens = self._read_line(self.tgt_file, self.tgt_offsets[idx].item())
        
        # Truncate
        if len(tgt_tokens) > self.max_len:
            tgt_tokens = tgt_tokens[:self.max_len]
            
        tgt = ['<sos>'] + tgt_tokens + ['<eos>']
        
        src_indices = torch.tensor(self.src_vocab.encode(src), dtype=torch.long)
        tgt_indices = torch.tensor(self.tgt_vocab.encode(tgt), dtype=torch.long)
        
        return src_indices, tgt_indices

class TranslationDataset(Dataset):
    """Dataset for translation tasks"""
    def __init__(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None, reverse_src=False, max_len=50):
        self.src_sentences = []
        self.tgt_sentences = []
        self.max_len = max_len
        
        # Read source file
        with open(src_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                # Apply HTML unescape
                tokens = [html.unescape(t) for t in tokens]
                if reverse_src:
                    tokens = list(reversed(tokens))
                self.src_sentences.append(tokens)
        
        # Read target file
        with open(tgt_file, 'r', encoding='utf-8') as f:
            for line in f: 
                tokens = line.strip().split()
                # Apply HTML unescape
                tokens = [html.unescape(t) for t in tokens]
                self.tgt_sentences.append(tokens)
        
        assert len(self.src_sentences) == len(self.tgt_sentences)
        
        # Build or use provided vocabularies
        # Build or use provided vocabularies
        if src_vocab is None:
            # Re-read or process stored sentences for vocab
            # Since tokens are already stored, we just flatten them
            # Note: We should assume stored tokens are already unescaped in __init__ reading loop below
            all_tokens = [tok for sent in self.src_sentences for tok in sent]
            limited_tokens = limit_vocab(all_tokens, max_size=50000)
            self.src_vocab = Vocab(limited_tokens)
        else:
            self.src_vocab = src_vocab
            
        if tgt_vocab is None:
            all_tokens = [tok for sent in self.tgt_sentences for tok in sent]
            limited_tokens = limit_vocab(all_tokens, max_size=50000)
            self.tgt_vocab = Vocab(limited_tokens)
        else:
            self.tgt_vocab = tgt_vocab
        
        # Cached lengths for bucket batching (computed lazily on first access)
        self._lengths_cache = None
    
    def compute_lengths(self):
        """
        Compute source sequence lengths for all examples efficiently.
        Uses already-loaded sentence lists.
        Returns tensor of lengths (including <eos> token).
        """
        if self._lengths_cache is not None:
            return self._lengths_cache
        
        # Compute lengths from stored sentences
        lengths = []
        for src_sent in self.src_sentences:
            # Apply max_len truncation
            token_count = min(len(src_sent), self.max_len)
            # Add 1 for <eos> token
            lengths.append(token_count + 1)
        
        self._lengths_cache = torch.tensor(lengths, dtype=torch.int64)
        return self._lengths_cache
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # 1. Source 처리: <sos> 제거
        # Encoder는 문장을 읽기만 하면 되므로 시작 토큰 불필요
        src_tokens = self.src_sentences[idx]
        if len(src_tokens) > self.max_len:
            src_tokens = src_tokens[:self.max_len]
        src = src_tokens + ['<eos>'] 
        
        # 2. Target 처리: 학습용 전체 시퀀스 생성
        tgt_tokens = self.tgt_sentences[idx]
        if len(tgt_tokens) > self.max_len:
            tgt_tokens = tgt_tokens[:self.max_len]
        tgt = ['<sos>'] + tgt_tokens + ['<eos>']
        
        src_indices = torch.tensor(self.src_vocab.encode(src), dtype=torch.long)
        tgt_indices = torch.tensor(self.tgt_vocab.encode(tgt), dtype=torch.long)
        return src_indices, tgt_indices
import torch

def left_pad_sequence(seqs, padding_value, batch_first=False):
    """
    seqs: list[Tensor] where each Tensor is shape (len,)
    returns: Tensor shape (max_len, batch) if batch_first=False else (batch, max_len)
    """
    assert len(seqs) > 0
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.int64)
    max_len = int(lengths.max().item())

    # dtype/device는 첫 샘플 기준(일반적으로 CPU long)
    out = seqs[0].new_full((len(seqs), max_len), fill_value=padding_value)  # (batch, max_len)

    for i, s in enumerate(seqs):
        l = s.size(0)
        out[i, max_len - l:] = s  # 오른쪽에 실제 토큰을 붙임 => 왼쪽이 pad

    if batch_first:
        return out, lengths
    else:
        return out.t().contiguous(), lengths  # (max_len, batch)
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_idx_src, pad_idx_tgt, left_pad_src=True, left_pad_tgt=False):
    src_batch, tgt_batch = zip(*batch)

    # lengths (패딩 전)
    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.int64)
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch], dtype=torch.int64)

    # pad
    if left_pad_src:
        src_padded, _ = left_pad_sequence(list(src_batch), padding_value=pad_idx_src, batch_first=False)
    else:
        src_padded = pad_sequence(src_batch, padding_value=pad_idx_src, batch_first=False)

    if left_pad_tgt:
        tgt_padded, _ = left_pad_sequence(list(tgt_batch), padding_value=pad_idx_tgt, batch_first=False)
    else:
        tgt_padded = pad_sequence(tgt_batch, padding_value=pad_idx_tgt, batch_first=False)

    return src_padded, tgt_padded, src_lengths, tgt_lengths
def dataset_factory(args, device):
    """
    WMT14/15 데이터셋 로더
    Returns both SRC and TGT metadata/vocab.
    
    Args:
        args: 학습 인자
            - args.dataset: 데이터셋 이름
                * 'wmt14-en-de': WMT14 English→German
                * 'wmt15-deen':  WMT15 German→English
                * 'sample100k': 샘플 데이터셋
            - args.reverse: Source 문장 역순 처리 여부 (동적)
            - args.batch_size: 배치 크기
        device: PyTorch device (CPU/GPU)
    
    Returns:
        src_metadata: Source 메타정보 (vocab_size, padding_idx 등)
        tgt_metadata: Target 메타정보 (vocab_size, padding_idx 등)
        src_vocab: Source 언어 Vocabulary
        tgt_vocab: Target 언어 Vocabulary
        train_iter: 학습 데이터 반복자
        val_iter:  검증 데이터 반복자
        test_iter: 테스트 데이터 반복자
    """
    print(f"Loading data for {args.dataset}...")

    # Determine dataset version
    if 'sample100k' in args.dataset. lower():
        root_dir = 'data/sample100k'
    elif 'author_data' in args.dataset.lower():
        root_dir = 'data/author_data'
    elif 'wmt15' in args.dataset.lower():
        root_dir = 'data/wmt15_vocab50k/base'
    else:
        root_dir = 'data/wmt14_vocab50k/base'
    
    data_dir = root_dir
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Determine translation direction
    if 'deen' in args.dataset.lower():
        src_ext, tgt_ext = 'de', 'en'  # German → English
        print("Direction: German → English")
    else:
        src_ext, tgt_ext = 'en', 'de'  # English → German
        print("Direction: English → German")

    # Check if we should reverse source sentences (runtime option)
    reverse_src = getattr(args, 'reverse', False)
    if reverse_src:
        print("Source sentence reversal:  ENABLED (dynamic)")
    
    # Load training data first to build vocabulary
    # Use LazyTranslationDataset for WMT datasets to save memory
    DatasetClass = LazyTranslationDataset if 'wmt' in args.dataset.lower() else TranslationDataset
    
    max_len = getattr(args, 'max_seq_len', 50)
    print(f"Max sequence length: {max_len}")

    
    # Determine filenames based on dataset type
    if 'author_data' in args.dataset.lower():
        train_src_name = f'train.10k.{src_ext}'
        train_tgt_name = f'train.10k.{tgt_ext}'
        val_src_name = f'valid.100.{src_ext}'
        val_tgt_name = f'valid.100.{tgt_ext}'
        test_src_name = f'test.100.{src_ext}'
        test_tgt_name = f'test.100.{tgt_ext}'
    else:
        train_src_name = f'train.{src_ext}'
        train_tgt_name = f'train.{tgt_ext}'
        val_src_name = f'valid.{src_ext}'
        val_tgt_name = f'valid.{tgt_ext}'
        test_src_name = f'test.{src_ext}'
        test_tgt_name = f'test.{tgt_ext}'

    train_dataset = DatasetClass(
        os.path.join(data_dir, train_src_name),
        os.path.join(data_dir, train_tgt_name),
        reverse_src=reverse_src,
        max_len=max_len
    )
    
    # Use training vocab for validation and test
    val_dataset = DatasetClass(
        os.path.join(data_dir, val_src_name),
        os.path.join(data_dir, val_tgt_name),
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        reverse_src=reverse_src,
        max_len=max_len
    )
    
    test_dataset = DatasetClass(
        os.path.join(data_dir, test_src_name),
        os.path.join(data_dir, test_tgt_name),
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        reverse_src=reverse_src,
        max_len=max_len
    )
    
    print(f"Vocab size: SRC={len(train_dataset.src_vocab)}, TGT={len(train_dataset.tgt_vocab)}")
    
    # Create DataLoaders
    # Note: assume shared specials so pad_idx is 0 for both; use TGT pad_idx for loss
    pad_idx_src = train_dataset.src_vocab.stoi.get('<pad>', 0)
    pad_idx_tgt = train_dataset.tgt_vocab.stoi.get('<pad>', 0)
    
    # Check if bucket batching is enabled
    use_bucket_batching = getattr(args, 'bucket_batching', False)
    
    if use_bucket_batching:
        print("Using length-bucketed batching for training data")
        # Compute lengths for bucket batching
        train_lengths = train_dataset.compute_lengths()
        
        # Create bucket batch sampler
        bucket_sampler = BucketBatchSampler(
            dataset_lengths=train_lengths,
            batch_size=args.batch_size,
            drop_last=False
        )
        
        # Create DataLoader with batch_sampler (cannot use batch_size or shuffle)
        train_iter = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt),
            num_workers=getattr(args, 'num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        # Standard DataLoader with shuffle
        train_iter = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Enable shuffling for standard training
            collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt),
            num_workers=getattr(args, 'num_workers', 0),
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    val_iter = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt),
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_iter = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt),
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Build SRC/TGT metadata separately
    # Note: Both vocabs have '<pad>' at index 0 by construction in Vocab class

    print('pad_idx_tgt', pad_idx_tgt)
    #assert pad_idx_src == 0 and pad_idx_tgt == 0, "Padding token must be at index 0"
    
    src_metadata = Metadata(vocab_size=len(train_dataset.src_vocab), padding_idx=pad_idx_tgt,src_padding_idx=pad_idx_src , vectors=None)
    tgt_metadata = Metadata(vocab_size=len(train_dataset.tgt_vocab), padding_idx=pad_idx_tgt,src_padding_idx=pad_idx_src, vectors=None)
    
    # Return both vocabularies and iterators
    return src_metadata, tgt_metadata, train_dataset.src_vocab, train_dataset.tgt_vocab, BatchWrapper(train_iter, device), BatchWrapper(val_iter, device), BatchWrapper(test_iter, device)

class Batch:
    """Wrapper for batch data"""
    def __init__(self, src, trg, src_lengths,tgt_lengths, device):
        self.src = src.to(device) if device else src
        self.trg = trg.to(device) if device else trg
        self.src_lengths = src_lengths.to(device) if device else src_lengths
        self.tgt_lengths = tgt_lengths.to(device) if device else tgt_lengths
        self.question = self.src
        self.answer = self.trg

class BatchWrapper:
    def __init__(self, dataloader, device=None):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for src, trg, src_lengths,tgt_lengths in self.dataloader:
            yield Batch(src, trg, src_lengths,tgt_lengths, self.device)
    
    def __len__(self):
        return len(self.dataloader)

# Field 생성을 위한 factory (기존 코드 호환용, 필요시 사용)
def field_factory(args):
    # Return a dummy object that won't be used
    return None

def metadata_factory(args, vocab):
    src_pad_idx=vocab.stoi.get('<pad>', 0)
    pad_idx = vocab.stoi.get('<pad>', 0)
    return Metadata(vocab_size=len(vocab), padding_idx=pad_idx, src_padding_idx=src_pad_idx, vectors=None)
