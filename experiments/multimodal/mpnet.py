import json
import logging
from typing import Dict, Literal, List, Union

import numpy as np
from safetensors import safe_open
from scipy.special import erf

logging.basicConfig(level=logging.INFO)


def load_safe_tensors(safetensors_path: str, mode: Literal['sentence_transformer', 'numpy']) -> Dict[str, np.ndarray]:
    tensors = {}
    f = safe_open(safetensors_path, framework='pt')
    for key in f.keys():
        if mode == 'numpy':
            tensors[key] = np.array(f.get_tensor(key))
        else:
            tensors[f'0.auto_model.{key}'] = f.get_tensor(key)
    return tensors


def load_json(file_name: str) -> Dict:
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data


def transpose_for_scores(x: np.ndarray, num_attention_heads: int = 12, attention_head_size: int = 64) -> np.ndarray:
    # Transform from (1,seq_len,embedding_dim) -> (1, num_attention_heads, seq_len, attention_head_size)
    new_x_shape = x.shape[:-1] + (num_attention_heads, attention_head_size)

    x = np.reshape(x, new_x_shape)

    return np.transpose(x, (0, 2, 1, 3))


class Linear:
    """
    y = x*W^T + b

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.

    Attributes
    ----------
    weights : np.ndarray
        The weight matrix of the layer (out_features, in_features).
    bias : np.ndarray
        The bias vector of the layer (out_features,).

    """

    def __init__(self, in_features: int, out_features: int):
        self.out_features = out_features
        self.in_features = in_features
        self.weights = np.random.randn(self.out_features, self.in_features)
        self.bias = np.zeros(self.out_features)

    def forward(self, input_tensor) -> np.ndarray:
        return np.matmul(input_tensor, self.weights.T) + self.bias


class LayerNorm:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Parameters
    ----------
    normalized_shape : int
        Size of the layer's output.
    beta : np.ndarray
        The shift parameter (bias) to be added.
    gamma : np.ndarray
        The scale parameter to be multiplied.
    eps : float
        A value added to the denominator for numerical stability (default: 1e-12).

    """

    def __init__(self, normalized_shape: int, beta: np.ndarray,
                 gamma: np.ndarray, eps: float = 1e-12):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.beta = beta
        self.gamma = gamma

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
         Apply the layer normalization on the input tensor.

         Parameters
         ----------
         x : np.ndarray
             The input data for normalization.

         Returns
         -------
         np.ndarray
             The normalized data.
         """

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        return (((x - mean) / np.sqrt(var + self.eps)) * self.gamma) + self.beta


class Embedding:
    """
        Turns positive integers (indexes) into dense vectors of fixed size.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        embedding_dim : int
            Dimension of the dense embedding.
        vocab : Dict
            A dictionary mapping tokens to their indices in the vocabulary.
        weights : np.ndarray
            Pretrained embedding weights.
     """

    def __init__(self, vocab_size: int, embedding_dim: int, vocab: Dict, weights: np.ndarray):
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = weights

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Maps indices (tokens) to their corresponding embeddings.

        Parameters
        ----------
        indices : np.ndarray
          Array of indices to be converted into embeddings.

        Returns
        -------
        np.ndarray
          Array of embeddings corresponding to the input indices.

        """
        valid_indices = np.where(indices < self.vocab_size, indices, self.vocab.get('<unk>'))
        return self.embeddings[valid_indices]


class Pooling:
    """
    The model is currently used for single sentences in future updates
    it will include other functionalities that's why it was left as a class
    """

    def forward(self, features: Dict) -> Dict:

        for i in ['token_embeddings', 'attention_mask']:
            if i not in list(features.keys()):
                raise TypeError('Wrong format!, input data should be in form of token_embeddings:embeddings \n'
                                'attention_mask : List[float]')

        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        output_vectors = []
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = input_mask_expanded.astype(float)
        np.sum(token_embeddings)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)

        sum_mask = input_mask_expanded.sum(axis=1)
        sum_mask = np.clip(sum_mask, 1e-9, np.inf)

        output_vectors.append(sum_embeddings / sum_mask)

        output_vector = np.concatenate(output_vectors, axis=1)
        features.update({'sentence_embedding': output_vector})
        return features


class Activation:
    @staticmethod
    def softmax(x) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def gelu(x, approximate=None) -> np.ndarray:
        if approximate:
            return 0.5 * x * (1 + Activation.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        else:
            return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    @staticmethod
    def tanh(x) -> np.ndarray:
        return np.tanh(x)


def normalize(x: np.ndarray, p: int = 2, dim: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    Applies normalization using Euclidean Norm
    """
    lp_norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)

    lp_norm = np.maximum(lp_norm, eps)

    normalized = x / lp_norm
    return normalized.astype(np.float32)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab: Dict, precomputed_map_path: str, max_input_chars_per_word=100,
                 bos_token="<s>",
                 eos_token="</s>",
                 sep_token="</s>",
                 cls_token="<s>",
                 unk_token="[UNK]",
                 pad_token="<pad>",
                 mask_token="<mask>"):
        self.vocab = vocab
        self.padding_idx = 1
        self.precomputed_map = load_json(precomputed_map_path)
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.index_to_token = {v: k for k, v in self.vocab.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token

    def tokenize(self, text: str) -> List[Union[int, np.ndarray]]:
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens.

        Returns:
            A list of wordpiece tokens.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        output_tokens = []
        for token in self.whitespace_tokenize(text):
            if token in list(self.precomputed_map.keys()):
                output_tokens.append(self.precomputed_map[token])
                continue

            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def encode(self, text):
        output_tokens = self.tokenize(text)
        if isinstance(output_tokens[0][0], list) and all(isinstance(i, float) for i in output_tokens[0][0]):
            return output_tokens[0]

        input_ids = []
        for token in output_tokens:
            if isinstance(token, str):
                token_id = self.vocab.get(token, self.vocab.get(self.unk_token))
                input_ids.append(token_id)

        input_ids = [self.vocab.get(self.bos_token)] + input_ids + [self.vocab.get(self.eos_token)]

        return input_ids

    def bml_tokenize(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            raise ValueError("text cannot be empty")

        tokens = self.whitespace_tokenize(text)
        embeddings = []

        for token in tokens:
            if token in self.precomputed_map:
                embeddings.extend(self.precomputed_map[token])
            else:
                embeddings.extend(self.encode(text))
                logging.info('token not found in precomputed_map, processing token in usual way')
        return embeddings

    @staticmethod
    def whitespace_tokenize(text: str) -> List[str]:
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        if isinstance(text, str):
            text = text.strip()

        if not text:
            return []
        tokens = text.split()
        return tokens


class InputEmbedding:
    """
      Represents the input embedding layer in the MPNet model.

      This class is responsible for creating word and position embeddings from the input tokens,
      and applies layer normalization to the combined embeddings.

      Parameters
      ----------
      hidden_size : int
          The size of each embedding vector.
      max_position_embeddings : int
          The maximum number of position embeddings.
      tokenizer : WordpieceTokenizer
          The tokenizer used to tokenize input text.
      vocab_size : int
          The size of the vocabulary.
      weights : Dict
          A dictionary containing the model's pretrained weights.
      layer_norm_eps : float
          Epsilon value for layer normalization.
    """

    tokenizer: WordpieceTokenizer
    weights: Dict
    word_embeddings: Embedding
    position_embeddings: Embedding
    layer_norm: LayerNorm

    def __init__(self, hidden_size: int, max_position_embeddings: int, tokenizer: WordpieceTokenizer, vocab_size: int,
                 weights: Dict, layer_norm_eps: float):

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")

        if not isinstance(max_position_embeddings, int) or max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be a positive integer")

        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")

        if not isinstance(weights, dict):
            raise TypeError("weights must be a dictionary")

        self.position_ids = None
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.weights = weights
        self.layer_norm_eps = layer_norm_eps

        self.word_embeddings = Embedding(self.vocab_size, self.hidden_size, self.tokenizer.vocab,
                                         self.weights['embeddings.word_embeddings.weight'])

        self.position_embeddings = Embedding(self.max_position_embeddings, self.hidden_size,
                                             self.tokenizer.vocab,
                                             self.weights['embeddings.position_embeddings.weight'])

        self.layer_norm = LayerNorm(hidden_size, eps=self.layer_norm_eps,
                                    gamma=self.weights['embeddings.LayerNorm.weight'],
                                    beta=self.weights['embeddings.LayerNorm.bias'])

        self.token_to_embedding = {token: self.word_embeddings.embeddings[index] for token, index in
                                   self.tokenizer.vocab.items()}

    def bml_encode(self, input_data: Union[str, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Encodes the given text into a list of embedding vectors.

        Parameters
        ----------
        input_data
           The input text to encode.

        Returns
        -------
        List[np.ndarray]
           A list of embedding vectors corresponding to the input text.
        """
        if isinstance(input_data, str):
            try:
                tokens = self.tokenizer.tokenize(input_data)
                return [self.token_to_embedding.get(self.tokenizer.bos_token)] + \
                       [self.token_to_embedding.get(i) if isinstance(i, str) else i for i in tokens] + \
                       [self.token_to_embedding.get(self.tokenizer.eos_token)]
            except Exception as e:
                raise RuntimeError(f'Failed to encode {input_data}, produces following error: {e}')
        elif isinstance(input_data, list) and all(isinstance(item, np.ndarray) for item in input_data):
            return input_data
        else:
            raise TypeError("input_data must be a string or a list of np.ndarray")

    def forward(self, input_data: Union[np.ndarray, List[np.ndarray]], position_ids: np.ndarray) -> np.ndarray:
        """
        Combines word and position embeddings and applies layer normalization.

        Parameters
        ----------
        input_data : np.ndarray
            The input token IDs.
        position_ids : np.ndarray
            The position IDs.

        Returns
        -------
        np.ndarray
            The normalized embeddings.
        """
        if isinstance(input_data, np.ndarray) and all(isinstance(i, np.int64) for i in input_data):
            inputs_embeds = np.expand_dims(self.word_embeddings.forward(input_data), axis=0)
        elif isinstance(input_data, np.ndarray) and all(isinstance(item, np.ndarray) for item in input_data):
            inputs_embeds = input_data
        else:
            raise TypeError(f"input_data must be an np.ndarray or a list of np.ndarray received type{input_data}")

        position_embeddings_res_ = self.position_embeddings.forward(position_ids)
        embeddings = inputs_embeds + position_embeddings_res_
        return self.layer_norm.forward(embeddings)

    def create_position_ids_from_input_ids(self, input_ids: np.ndarray, padding_idx: int) -> np.ndarray:
        """
        Creates position IDs from input token IDs.

        Parameters
        ----------
        input_ids : np.ndarray
            The input token IDs.
        padding_idx : int
            The index used for padding.

        Returns
        -------
        np.ndarray
            The generated position IDs.
        """
        if isinstance(input_ids, np.ndarray) and all(isinstance(item, np.ndarray) for item in input_ids):
            position_ids = np.arange(len(input_ids)) + 2
            for index, item in enumerate(input_ids):
                if np.array_equal(item, self.tokenizer.padding_idx):
                    position_ids[index] = padding_idx
            return position_ids
        else:
            mask = (input_ids != padding_idx).astype(int)
            incremental_indices = np.cumsum(mask, axis=0) * mask
            return incremental_indices + padding_idx


class MPNetAttention:
    """
        Implements the attention mechanism for MPNet.

        This class is responsible for computing the attention scores and applying
        them to the input hidden states, following the architecture of MPNet. It
        involves multiple linear layers for query (q), key (k), value (v), and output
        (o), along with layer normalization and intermediate dense layers.

        Parameters
        ----------
        num_hidden_layers : int
            The number of hidden layers in the attention mechanism.
        all_head_size : int
            Total size of all attention heads.
        hidden_size : int
            The size of the hidden layers.
        intermediate_size : int
            The size of the intermediate layers.
        weights : Dict
            A dictionary containing the weights for various layers.
        attention_head_size : int
            The size of each attention head.
    """

    def  __init__(self, num_hidden_layers: int, all_head_size: int, hidden_size: int, intermediate_size: int,
                 weights: Dict, attention_head_size: int):

        self.num_hidden_layers = num_hidden_layers
        self.all_head_size = all_head_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weights = weights
        self.attention_head_size = attention_head_size
        self.q_linears = [Linear(self.all_head_size, self.all_head_size) for _ in range(12)]
        self.k_linears = [Linear(self.all_head_size, self.all_head_size) for _ in range(12)]
        self.v_linears = [Linear(self.all_head_size, self.all_head_size) for _ in range(12)]
        self.o_linears = [Linear(self.all_head_size, self.all_head_size) for _ in range(12)]

        self.layer_norm = [LayerNorm(self.hidden_size, gamma=np.ones(self.hidden_size), beta=np.zeros(self.hidden_size))
                           for _ in range(12)]

        self.output_dense = [Linear(self.intermediate_size, self.hidden_size) for _ in range(12)]

        self.intermediate_dense = [Linear(self.hidden_size, self.intermediate_size) for _ in range(12)]

        self.output_layernorm = [LayerNorm(self.hidden_size, gamma=np.ones(self.hidden_size),
                                           beta=np.zeros(self.hidden_size)) for _ in range(12)]

    def forward(self, hidden_states: np.ndarray, relative_position_bias: np.ndarray,
                extended_attention_mask: np.ndarray) -> np.ndarray:
        """

        This method applies the attention mechanism to the input hidden states using
        the provided relative position bias and extended attention mask. It sequentially
        processes the data through multiple layers of the network.

        Parameters
        ----------
        hidden_states : np.ndarray
            The hidden states input tensor of shape (batch_size, seq_length, hidden_size).
        relative_position_bias : np.ndarray
            Tensor containing relative position bias.
        extended_attention_mask : np.ndarray
            Tensor representing the extended attention mask.

        Returns
        -------
        np.ndarray
            The output tensor after applying the attention mechanism, of the same shape as `hidden_states`.
        """

        for layer_idx in range(self.num_hidden_layers):
            q_linear = self.q_linears[layer_idx]
            q_linear.weights = self.weights[f'encoder.layer.{layer_idx}.attention.attn.q.weight']
            q_linear.bias = self.weights[f'encoder.layer.{layer_idx}.attention.attn.q.bias']

            k_linear = self.k_linears[layer_idx]
            k_linear.weights = self.weights[f'encoder.layer.{layer_idx}.attention.attn.k.weight']
            k_linear.bias = self.weights[f'encoder.layer.{layer_idx}.attention.attn.k.bias']

            v_linear = self.v_linears[layer_idx]
            v_linear.weights = self.weights[f'encoder.layer.{layer_idx}.attention.attn.v.weight']
            v_linear.bias = self.weights[f'encoder.layer.{layer_idx}.attention.attn.v.bias']

            o_linear = self.o_linears[layer_idx]
            o_linear.weights = self.weights[f'encoder.layer.{layer_idx}.attention.attn.o.weight']
            o_linear.bias = self.weights[f'encoder.layer.{layer_idx}.attention.attn.o.bias']

            layer_norm = self.layer_norm[layer_idx]
            layer_norm.gamma = self.weights[f'encoder.layer.{layer_idx}.attention.LayerNorm.weight']
            layer_norm.beta = self.weights[f'encoder.layer.{layer_idx}.attention.LayerNorm.bias']

            intermediate_dense = self.intermediate_dense[layer_idx]
            intermediate_dense.weights = self.weights[f'encoder.layer.{layer_idx}.intermediate.dense.weight']
            intermediate_dense.bias = self.weights[f'encoder.layer.{layer_idx}.intermediate.dense.bias']

            output_dense = self.output_dense[layer_idx]
            output_dense.weights = self.weights[f'encoder.layer.{layer_idx}.output.dense.weight']
            output_dense.bias = self.weights[f'encoder.layer.{layer_idx}.output.dense.bias']

            output_layernorm = self.output_layernorm[layer_idx]
            output_layernorm.gamma = self.weights[f'encoder.layer.{layer_idx}.output.LayerNorm.weight']
            output_layernorm.beta = self.weights[f'encoder.layer.{layer_idx}.output.LayerNorm.bias']

            q_ = q_linear.forward(hidden_states)
            k_ = k_linear.forward(hidden_states)
            v_ = v_linear.forward(hidden_states)

            q = transpose_for_scores(q_)  # (Batch, num_attention_heads, seq_len, attention_head_size)
            k = transpose_for_scores(k_)
            v = transpose_for_scores(v_)

            attention_scores = np.matmul(q, np.transpose(k, (0, 1, -1, -2)))

            attention_scores = attention_scores / np.sqrt(
                self.attention_head_size)  # (Batch, num_attention_heads, seq_len, seq_len )

            if relative_position_bias is not None:
                attention_scores += relative_position_bias

            if extended_attention_mask is not None:
                attention_scores = attention_scores + extended_attention_mask

            attention_probs = Activation.softmax(attention_scores)

            context_layer = np.matmul(attention_probs, v)  # (Batch, num_attention_heads,seq_len,attention_head_size)

            context_layer = np.transpose(context_layer,
                                         (0, 2, 1, 3))  # (Batch, seq_len, num_attention_heads, attention_head_size)
            new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
            context_layer = np.reshape(context_layer, new_context_layer_shape)  # (1, seq_len, emb_dim)
            o = o_linear.forward(context_layer)  # (Batch, seq_len, emb_dim)

            attention_output = layer_norm.forward(o + hidden_states)

            intermediate_output = intermediate_dense.forward(attention_output)  # (Batch, seq_len, intermediate_size)
            intermediate_output = Activation.gelu(intermediate_output, approximate=None)

            layer_output = output_dense.forward(intermediate_output)  # (Batch, seq_len, emb_dim)

            current_layer_output = output_layernorm.forward(
                layer_output + attention_output)

            hidden_states = current_layer_output  # (Batch, seq_len, emb_dim)

        return hidden_states


class MPNetEncoder:
    """
    Represents the encoder part of the MPNet model.

    This class encapsulates the attention mechanism of MPNet along with the computation of relative position bias.
    It processes input embeddings through multiple layers of attention and normalization.

    Parameters
    ----------
    hidden_size : int
        The size of the hidden layers.
    weights : Dict
        A dictionary containing the weights for various layers.
    all_head_size : int
        Total size of all attention heads.
    intermediate_size : int
        The size of the intermediate layers.
    num_hidden_layers : int
        The number of hidden layers in the attention mechanism.
    attention_head_size : int
        The size of each attention head.
    relative_attention_num_buckets : int
        Number of buckets for relative attention.
    num_attention_heads : int
        The number of attention heads.
    tokenizer : WordpieceTokenizer
        The tokenizer used for processing input text.

    """

    attention: MPNetAttention
    relative_attention_bias: Embedding

    def __init__(self, hidden_size: int, weights: Dict, all_head_size: int, intermediate_size: int,
                 num_hidden_layers: int, attention_head_size: int,
                 relative_attention_num_buckets: int, num_attention_heads: int, tokenizer: WordpieceTokenizer):

        self.hidden_size = hidden_size
        self.weights = weights
        self.all_head_size = all_head_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_size = attention_head_size
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.num_attention_heads = num_attention_heads
        self.tokenizer = tokenizer

        self.attention = MPNetAttention(self.num_hidden_layers, self.all_head_size, self.hidden_size,
                                        self.intermediate_size, self.weights, self.attention_head_size)

        self.relative_attention_bias = Embedding(self.relative_attention_num_buckets, self.num_attention_heads,
                                                 self.tokenizer.vocab,
                                                 self.weights['encoder.relative_attention_bias.weight'])

    def forward(self, input_embs: np.ndarray, position_ids: np.ndarray, extended_attention_mask: np.ndarray,
                pooling: bool = False) -> np.ndarray:

        """
        Processes the input embeddings through the MPNet encoder.

        Parameters
        ----------
        input_embs : np.ndarray
            The input embeddings.
        position_ids : np.ndarray
            The position IDs for the embeddings.
        extended_attention_mask : np.ndarray
            The attention mask for the embeddings.
        pooling : bool(default = False).

            Determines whether pooling is applied to the output
        Returns
        -------
        np.ndarray
            The processed embeddings after passing through the MPNet encoder.
        """

        position_bias = self.compute_position_bias(input_embs, position_ids)
        hidden_states = self.attention.forward(input_embs, position_bias, extended_attention_mask)

        if not pooling:
            return hidden_states[0]
        else:
            sequence_output = hidden_states[0]
            pooler_output = self.pooling.forward(sequence_output)

            return Activation.tanh(pooler_output)

    def compute_position_bias(self, x: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """
        Computes the position bias for the given inputs.

        Parameters
        ----------
        x : np.ndarray
            The input embeddings.
        position_ids : np.ndarray
            The position IDs for the embeddings.

        Returns
        -------
        np.ndarray
            The computed position bias.
        """

        bsz, qlen, klen = 1, x.shape[1], x.shape[1]
        context_position = position_ids[:, :, None]
        memory_position = position_ids[:, None, :]
        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(relative_position)
        values = self.relative_attention_bias.forward(rp_bucket)
        values = np.transpose(values, axes=(0, 3, 1, 2))
        return np.broadcast_to(values, (bsz, values.shape[1], qlen, klen))

    @staticmethod
    def relative_position_bucket(relative_position: np.ndarray, num_buckets=32, max_distance=128) -> np.ndarray:
        """
        Computes relative position buckets for the given relative positions.

        Parameters
        ----------
        relative_position : np.ndarray
            The relative positions to bucketize.
        num_buckets : int, optional
            The number of buckets (default is 32).
        max_distance : int, optional
            The maximum distance for bucket computation (default is 128).

        Returns
        -------
        np.ndarray
            The bucketized relative positions.
        """

        num_buckets //= 2
        n = -relative_position
        ret = np.where(n < 0, num_buckets, 0)
        n = np.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                np.log(n / max_exact) / np.log(max_distance / max_exact) * (num_buckets - max_exact)).astype(int)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret


class MPNetModel:
    """
        A model class that implements the MPNet architecture for natural language processing tasks.

        This class encapsulates the MPNet model, including its tokenizer, input embedding, encoder, and pooling layers.
        It is designed to process text input and produce corresponding sentence embeddings.

        Parameters
        ----------
        config : Dict
            Configuration dictionary containing model parameters like vocab size, hidden size, etc.
        weights_dict : str
            Path to the file containing the pre-trained weights of the model.
        tokenizer_config : Dict
            Configuration dictionary for the tokenizer.

        Attributes
        ----------
        attention_head_size : int
            Size of each attention head in the model.
        padding_idx : int
            Index used for padding sequences.
        vocab_size : int
            Size of the vocabulary.
        hidden_size : int
            Dimensionality of the hidden layers.
        num_hidden_layers : int
            Number of hidden layers in the encoder.
        num_attention_heads : int
            Number of attention heads in the model.
        hidden_act : str
            Activation function used in hidden layers.
        intermediate_size : int
            Size of intermediate layers in the model.
        hidden_dropout_prob : float
            Dropout probability for hidden layers.
        attention_probs_dropout_prob : float
            Dropout probability for attention probabilities.
        max_position_embeddings : int
            Maximum number of position embeddings.
        initializer_range : float
            Range of the initializer.
        layer_norm_eps : float
            Epsilon value for layer normalization.
        relative_attention_num_buckets : int
            Number of buckets for relative attention.
        all_head_size : int
            Combined size of all attention heads.
        model_weights : Dict
            Loaded model weights.
        tokenizer : WordpieceTokenizer
            Tokenizer used for processing input text.
        input_embedding : InputEmbedding
            Layer to embed input sequences.
        encoder : MPNetEncoder
            MPNet encoder for processing embedded sequences.
        pooler : Pooling
            Pooling layer for sentence embeddings.
        """

    tokenizer: WordpieceTokenizer
    vocab: Dict
    input_embedding: InputEmbedding
    encoder: MPNetEncoder
    pooler: Pooling

    def __init__(self, config: Dict, weights_dict: str, tokenizer_config: Dict):
        self.attention_head_size = 64
        self.padding_idx = 1

        if not isinstance(config, Dict):
            raise ValueError("config must be a dictionary")

        if not isinstance(tokenizer_config, Dict):
            raise ValueError("tokenizer_config must be a dictionary")

        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_act = config['hidden_act']
        self.intermediate_size = config['intermediate_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.max_position_embeddings = config['max_position_embeddings']
        self.initializer_range = config['initializer_range']
        self.layer_norm_eps = config['layer_norm_eps']
        self.relative_attention_num_buckets = config['relative_attention_num_buckets']
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        try:
            self.model_weights = load_safe_tensors(weights_dict, mode='numpy')
        except FileNotFoundError:
            raise FileNotFoundError(f"Weight file not found at {weights_dict}")

        self.tokenizer = WordpieceTokenizer(vocab=tokenizer_config['model']['vocab'],
                                            precomputed_map_path='google_10k_embs.json')
        self.input_embedding = InputEmbedding(self.hidden_size, self.max_position_embeddings, self.tokenizer,
                                              self.vocab_size, self.model_weights, self.layer_norm_eps)

        self.encoder = MPNetEncoder(self.hidden_size, self.model_weights, self.all_head_size, self.intermediate_size,
                                    self.num_hidden_layers, self.attention_head_size,
                                    self.relative_attention_num_buckets, self.num_attention_heads, self.tokenizer)
        self.pooler = Pooling()

    def forward(self, text: str) -> np.ndarray:

        """
        Processes the input text and returns the corresponding sentence embedding.

        Parameters
        ----------
        text : str
            The input text to process.

        Returns
        -------
        np.ndarray
            The final output from recreated sentence-transformer model
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text:
            raise ValueError("text cannot be empty")

        try:
            input_ids = np.array(self.tokenizer.bml_tokenize(text))

            attention_mask = np.ones((1, len(input_ids)))

            extended_attention_mask = np.expand_dims(np.expand_dims(attention_mask, axis=1), axis=1)
            extended_attention_mask = (1.0 - extended_attention_mask) * np.finfo(np.float32).min

            position_ids = np.expand_dims(
                self.input_embedding.create_position_ids_from_input_ids(input_ids, self.padding_idx), axis=0)
            input_embedding_output = self.input_embedding.forward(input_ids, position_ids=position_ids)

            encoder_output = self.encoder.forward(input_embedding_output, position_ids, extended_attention_mask)
            features = {'token_embeddings': encoder_output, 'attention_mask': attention_mask}

            pooling_output = self.pooler.forward(features)
            final_embedding = normalize(pooling_output['sentence_embedding'], p=2, dim=1, eps=1e-12)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")

        return final_embedding


if __name__ == "__main__":
    ...

    # load_dotenv()
    # config_path = load_json(os.getenv('CONFIG_PATH'))
    # weights_path = os.getenv('WEIGHTS_PATH')
    # tokenizer_path = load_json(os.getenv('TOKENIZER_CONFIG_PATH'))

    # mpnet = MPNetModel(config=config_path,
    #                    weights_dict=weights_path, tokenizer_config=tokenizer_path)
    #
    # inference_times = []
    # for j in range(100):
    #     start_time = time.time()
    #     encoded_embedding = mpnet.forward('school')
    #     end_time = time.time()
    #     inference_times.append(end_time - start_time)
    #
    # average_time = sum(inference_times) / len(inference_times)
    # print(f"Average Inference Time: {round(average_time, 2)} seconds")