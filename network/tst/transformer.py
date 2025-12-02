import torch
import torch.nn as nn

from network.tst.encoder import Encoder
from network.tst.decoder import Decoder
from network.tst.utils import generate_original_PE, generate_regular_PE


class TransformerFea(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        channel: int,
        q: int,
        v: int,
        h: int,
        N: int,
        attention_size: int = None,
        dropout: float = 0.3,
        chunk_mode: str = "chunk",
        pe: str = None,
        pe_period: int = 24,
    ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList(
            [
                Encoder(
                    d_model,
                    q,
                    v,
                    h,
                    attention_size=attention_size,
                    dropout=dropout,
                    chunk_mode=chunk_mode,
                )
                for _ in range(N)
            ]
        )
        self.layers_decoding = nn.ModuleList(
            [
                Decoder(
                    d_model,
                    q,
                    v,
                    h,
                    attention_size=attention_size,
                    dropout=dropout,
                    chunk_mode=chunk_mode,
                )
                for _ in range(N)
            ]
        )

        self._embedding = nn.Linear(d_input, d_model)
        self.in_features = d_model * channel

        pe_functions = {
            "original": generate_original_PE,
            "regular": generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.'
            )

        self.name = "transformerfea"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        x = torch.squeeze(x)
        K = x.shape[1]

        # Embeddin module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {"period": self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        # print(encoding.shape)
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            # print(encoding.shape)
        # Decoding stack
        decoding = encoding.view(encoding.shape[0], -1)

        return decoding


class TransformerFeaLAG(nn.Module):
    def __init__(
        self,
        args,
        d_model: int,
        q: int,
        v: int,
        h: int,
        N: int,
        attention_size: int = None,
        dropout: float = 0.3,
        chunk_mode: str = "chunk",
        pe: str = None,
        pe_period: int = 24,
    ):
        super().__init__()

        self._d_model = d_model
        self.patch_nums = args.grid_size
        self.layers_encoding = nn.ModuleList(
            [
                Encoder(
                    d_model,
                    q,
                    v,
                    h,
                    attention_size=attention_size,
                    dropout=dropout,
                    chunk_mode=chunk_mode,
                )
                for _ in range(N)
            ]
        )

        self._embedding = nn.Linear(args.input_shape[-1], d_model)
        self.in_features = d_model * args.input_shape[0]

        pe_functions = {
            "original": generate_original_PE,
            "regular": generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.'
            )

        self.name = "transformerfea"

        self.layers_encoding1 = nn.ModuleList(
            [
                Encoder(
                    d_model,
                    q,
                    v,
                    h,
                    attention_size=attention_size,
                    dropout=dropout,
                    chunk_mode=chunk_mode,
                )
                for _ in range(N)
            ]
        )
        self._embedding1 = nn.Linear(
            args.input_shape[-1] // args.grid_size * args.input_shape[0], d_model
        )
        self.in_features1 = d_model * args.grid_size

    def init(self, x):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x)
        K = x.shape[1]

        encoding = self._embedding(x)

        if self._generate_PE is not None:
            pe_params = {"period": self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        for layer in self.layers_encoding:
            encoding = layer(encoding)

        decoding = encoding.view(encoding.shape[0], -1)

        x1 = x.view(-1, x.shape[1], self.patch_nums, int(x.shape[2] / self.patch_nums))
        x1 = x1.permute(0, 2, 1, 3).contiguous()
        x1 = x1.view(x1.shape[0], x1.shape[1], -1)
        encoding1 = self._embedding1(x1)
        for layer in self.layers_encoding1:
            encoding1 = layer(encoding1)
        decoding1 = encoding1.view(encoding1.shape[0], -1)

        return decoding, decoding1
