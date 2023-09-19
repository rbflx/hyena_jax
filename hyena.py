import jax
from jax import numpy as jnp
from flax import linen as nn

from einops import rearrange


def fftconv(seq, filter, bias):
    """Batched convolution of seq and filter using FFT.
    Args:
        seq: Input sequence. (b, l, d)
        filter: Filter to convolve with. (l, d)
        bias: Bias to multiply with seq. (d,)
    """
    seq_len = seq.shape[-2]

    # Pad sequence and filter to be able to slice result up to seq_len (see below).
    fft_size = 2*seq_len
    filter_f = jnp.fft.rfft(filter, n=fft_size, axis=-2) / fft_size
    seq_f = jnp.fft.rfft(seq, n=fft_size, axis=-2)

    # Discard results after seq_len to preserve causality.
    y = jnp.fft.irfft(seq_f * filter_f, n=fft_size, norm='forward', axis=-2)[..., :seq_len, :]

    out = y + seq * bias

    return out


class ExponentialModulation(nn.Module):
    """Exponentially modulate a matrix along its temporal/sequence dimension."""
    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    shift: float = 0.0

    @nn.compact
    def __call__(self, t, x):
        """Modulate x using time/sequence values in t.
        Args:
            t: time/position values to modulate with. (l, 1) 
            x: input. (..., l, d)
        """
        in_dim = x.shape[-1]
        max_decay = jnp.log(self.target) / self.fast_decay_pct
        min_decay = jnp.log(self.target) / self.slow_decay_pct
        deltas = jnp.linspace(min_decay, max_decay, in_dim)     # (d,)

        decay = jnp.exp(-t * jnp.abs(deltas))                   # (l, d)
        out = x * (decay + self.shift)
        return out


class HyenaFilter(nn.Module):
    """Create implicit filters from positional embeddings."""
    mlp_width: int          # Width of implicit MLP.
    layers: int             # Number of filter layers.
    n_filters: int          # Number of filters to create. Usually (order-1)*d
    init_freq: float = 1.0  # Frequency to initialize sine activation with.

    @nn.compact
    def __call__(self, embeds):
        k = nn.Dense(self.mlp_width, name='filter_in')(embeds)

        freq = self.param(
            'freq',
            lambda rng, init_freq: jnp.full(self.mlp_width, init_freq),
            self.init_freq
        )

        k = jnp.sin(freq * k)

        for _ in range(self.layers):
            k = nn.Dense(self.mlp_width)(k)
            k = jnp.sin(freq * k)
        
        k = nn.Dense(self.n_filters, use_bias=False, name='filter_out')(k)  # (l, (o-1)*d)

        return k


class PosEmbeddings(nn.Module):
    """Create positional embeddings for a sequence of a certain maximum length."""
    max_len: int        # Maximum length of the sequence.
    pos_embed_dim: int  # Positional embedding dimension.

    @nn.compact
    def __call__(self, l):
        t = jnp.linspace(0, 1, self.max_len)[:l, None]          # (l, 1)

        # Initialization function for positional embeddings.
        def z_init(rng, l, t):
            assert self.pos_embed_dim % 2 == 1, "Positional embedding dimension must be odd (1 + real + imag)."

            bands = (self.pos_embed_dim - 1) // 2
            t_rescaled = jnp.linspace(0, self.max_len-1, self.max_len)[:l, None]   # (l, 1)
            w = 2 * jnp.pi * t_rescaled / l
            
            f = jnp.linspace(1e-4, bands-1, bands)[None]        # (1, bands)
            z = jnp.exp(-1j * f * w)                            # (l, bands)
            z = jnp.concatenate([t, z.real, z.imag], axis=-1)   # (l, 1+pos_embed_dim)

            return z

        z = self.param(
            'z',
            z_init,
            l, t
        )

        return z, t


class HyenaOperator(nn.Module):
    """Apply a Hyena operator to a given sequence."""
    max_len: int            # Maximum sequence length.
    d_model: int            # Width of Hyena layer.
    pos_embed_dim: int      # Position embedding dimension.
    filter_features: int    # Implicit filter dimension.
    num_filter_layers: int  # Number of filter creation layers.
    order: int = 2          # Depth of Hyena recurrence.
    init_freq: float = 1.0  # Initial sine activation frequency.
    dropout: float = 0.0    # Dropout rate.
    
    @nn.compact
    def __call__(self, u, train=True):
        # u: (b, l, embed_dim)

        l = min(u.shape[-2], self.max_len)

        z, t = PosEmbeddings(self.max_len, self.pos_embed_dim)(l)   # z: (l, 1+pos_embed_dim), t: (l, 1)
        n_filters = (self.order-1)*self.d_model
        filters = HyenaFilter(self.filter_features, self.num_filter_layers, n_filters, self.init_freq)(z) # (l, (o-1)*d)
        
        filters = rearrange(filters, 'l (o d) -> o l d', o=self.order-1)
        filters = ExponentialModulation()(t, filters)

        # number of projections = self.order * x and one v
        n_projs = (self.order+1)*self.d_model
        u = nn.Dense(n_projs)(u)                # (b, l, (o+1)*d)
        
        # depthwise 1d conv
        uc = nn.Conv(
            n_projs,
            kernel_size=(3,),
            padding='CAUSAL',
            feature_group_count=n_projs,
        )(u)
        
        *x, v = jnp.split(uc, self.order+1, axis=-1)    # o * (b, l, d), (b, l, d)

        # learned bias
        bias = self.param(
            'bias',
            nn.initializers.normal(stddev=1),   # stddev=1 to replicate torch init
            (self.order-1, self.d_model)
        )

        # Sequential application of pointwise multiplication with projection and convolution with implicit filter.
        for o, k_i in enumerate(filters):
            v = v * x[o]
            v = nn.Dropout(self.dropout)(v, deterministic=not train)
            v = fftconv(v, k_i, bias[o])

        y = v * x[-1]
        y = nn.Dropout(self.dropout)(y, deterministic=not train)

        y = nn.Dense(self.d_model)(y)

        return y