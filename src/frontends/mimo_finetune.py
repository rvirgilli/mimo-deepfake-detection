"""
Fine-tuning strategies for MiMo encoder.

This module provides parameter-efficient fine-tuning methods for the
MiMo-Audio-Tokenizer encoder, enabling stable training without catastrophic forgetting.

Strategies:
- Adapter: Bottleneck adapters inserted after transformer blocks (~2M params)
- LoRA: Low-rank adaptation of attention layers (~1M params)
- Partial: Fine-tune only last N transformer layers
- LayerwiseLR: All layers trainable with decaying learning rates

Usage:
    from src.frontends.mimo_finetune import apply_finetune_strategy

    # Apply adapter strategy
    frontend = apply_finetune_strategy(
        mimo_frontend,
        strategy='adapter',
        adapter_dim=64,
        adapter_layers='last_n',
        n_adapter_layers=8
    )

    # For optimizer, get trainable params:
    trainable_params = frontend.get_trainable_params()
"""

import math
from typing import Optional, List, Dict, Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor


class AdapterLayer(nn.Module):
    """
    Bottleneck adapter layer for parameter-efficient fine-tuning.

    The adapter uses a down-projection, non-linearity, and up-projection
    with a residual connection. Initialized to near-identity so the
    adapter starts as a passthrough.

    Architecture:
        input (d) -> down_proj (d -> bottleneck) -> GELU -> up_proj (bottleneck -> d) -> + input

    Args:
        dim: Input/output dimension (1280 for MiMo)
        bottleneck_dim: Bottleneck dimension (default: 64)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim

        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize to near-identity (adapter starts as passthrough)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch*seq_len, dim)
        Returns:
            Output tensor of same shape as input
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return residual + x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning.

    Decomposes weight updates into low-rank matrices: W' = W + BA
    where B is (out_features, rank) and A is (rank, in_features).

    Args:
        original_layer: The original nn.Linear layer to adapt
        rank: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor (default: 16)
        dropout: Dropout rate for LoRA (default: 0.0)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

        # Low-rank decomposition matrices - match dtype and device of original layer
        # A: (rank, in_features), B: (out_features, rank)
        dtype = original_layer.weight.dtype
        device = original_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor
        Returns:
            Original output + scaled low-rank update
        """
        original_out = self.original(x)
        # x @ A^T @ B^T = x @ (BA)^T
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return original_out + lora_out * self.scaling

    def num_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


class MiMoWithAdapters(nn.Module):
    """
    MiMo encoder wrapper with adapter layers for parameter-efficient fine-tuning.

    Adapters are inserted after transformer blocks. The base encoder is frozen,
    and only the adapters are trained.

    Args:
        tokenizer: MiMo tokenizer instance
        adapter_dim: Bottleneck dimension for adapters (default: 64)
        adapter_dropout: Dropout rate in adapters (default: 0.1)
        adapter_layers: Which layers get adapters - 'all', 'last_n', 'every_n' (default: 'last_n')
        n_adapter_layers: Number of layers for 'last_n' or 'every_n' mode (default: 8)
    """

    def __init__(
        self,
        tokenizer,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        adapter_layers: str = 'last_n',
        n_adapter_layers: int = 8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.adapter_dim = adapter_dim

        # Freeze base model
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # Get encoder hidden dimension
        self.hidden_dim = tokenizer.config.d_model  # 1280

        # Count transformer layers
        self.num_layers = self._count_layers()
        print(f"MiMo encoder has {self.num_layers} transformer layers")

        # Determine which layers get adapters
        if adapter_layers == 'all':
            adapter_indices = list(range(self.num_layers))
        elif adapter_layers == 'last_n':
            n = min(n_adapter_layers, self.num_layers)
            adapter_indices = list(range(self.num_layers - n, self.num_layers))
        elif adapter_layers == 'every_n':
            step = max(1, self.num_layers // n_adapter_layers)
            adapter_indices = list(range(0, self.num_layers, step))
        elif adapter_layers == 'first_n':
            n = min(n_adapter_layers, self.num_layers)
            adapter_indices = list(range(n))
        else:
            raise ValueError(f"Unknown adapter_layers: {adapter_layers}")

        print(f"Adding adapters to layers: {adapter_indices}")

        # Create adapters
        self.adapters = nn.ModuleDict({
            str(i): AdapterLayer(self.hidden_dim, adapter_dim, adapter_dropout)
            for i in adapter_indices
        })
        self.adapter_indices = set(adapter_indices)

        # Match adapter dtype and device to encoder
        encoder_param = next(self.tokenizer.encoder.parameters())
        self.adapters = self.adapters.to(dtype=encoder_param.dtype, device=encoder_param.device)

        # Register hooks
        self._hooks = []
        self._register_adapter_hooks()

        # Print param count
        total_adapter_params = sum(a.num_params() for a in self.adapters.values())
        print(f"Total adapter parameters: {total_adapter_params:,} ({total_adapter_params/1e6:.2f}M)")

    def _count_layers(self) -> int:
        """Count transformer layers in the encoder."""
        # Try to find the transformer layers
        encoder = self.tokenizer.encoder

        # Check common patterns
        for name, module in encoder.named_modules():
            if hasattr(module, 'layers') and isinstance(module.layers, nn.ModuleList):
                return len(module.layers)
            if hasattr(module, 'blocks') and isinstance(module.blocks, nn.ModuleList):
                return len(module.blocks)

        # Fallback: count modules with 'layer' or 'block' in name
        layer_count = 0
        for name, _ in encoder.named_modules():
            if '.layers.' in name or '.blocks.' in name:
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part in ('layers', 'blocks') and i + 1 < len(parts):
                        try:
                            idx = int(parts[i + 1])
                            layer_count = max(layer_count, idx + 1)
                        except ValueError:
                            pass

        return layer_count if layer_count > 0 else 32  # Default assumption

    def _register_adapter_hooks(self) -> None:
        """Register forward hooks to inject adapters after transformer layers."""
        encoder = self.tokenizer.encoder

        def make_hook(adapter_idx: int):
            def hook(module, input, output):
                if str(adapter_idx) in self.adapters:
                    # Handle tuple output (hidden_states, ...)
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        adapted = self.adapters[str(adapter_idx)](hidden_states)
                        return (adapted,) + output[1:]
                    else:
                        return self.adapters[str(adapter_idx)](output)
                return output
            return hook

        # Find transformer layers and register hooks
        for name, module in encoder.named_modules():
            for i in self.adapter_indices:
                # Match patterns like 'layers.0', 'blocks.0', 'encoder.layer.0'
                if (f'layers.{i}' in name or f'blocks.{i}' in name or f'layer.{i}' in name):
                    # Only hook the top-level layer module
                    if name.endswith(f'.{i}') or name.endswith(f'layers.{i}') or name.endswith(f'blocks.{i}'):
                        h = module.register_forward_hook(make_hook(i))
                        self._hooks.append(h)
                        print(f"  Registered adapter hook on: {name}")
                        break

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return only trainable (adapter) parameters."""
        return self.adapters.parameters()

    def num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.adapters.parameters())

    def forward(self, *args, **kwargs):
        """Forward pass through the tokenizer (adapters are applied via hooks)."""
        return self.tokenizer(*args, **kwargs)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


class MiMoWithLoRA(nn.Module):
    """
    MiMo encoder wrapper with LoRA for parameter-efficient fine-tuning.

    LoRA is applied to specified linear layers (typically attention projections).
    The base encoder is frozen, and only LoRA parameters are trained.

    Args:
        tokenizer: MiMo tokenizer instance
        lora_rank: Rank of low-rank decomposition (default: 8)
        lora_alpha: Scaling factor (default: 16)
        lora_dropout: Dropout rate (default: 0.0)
        target_modules: List of module name patterns to apply LoRA to
            (default: ['q_proj', 'v_proj'])
    """

    def __init__(
        self,
        tokenizer,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if target_modules is None:
            target_modules = ['q_proj', 'v_proj', 'query', 'value']

        # Freeze base model
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # Track LoRA layers
        self.lora_layers: Dict[str, LoRALayer] = {}

        # Apply LoRA to target modules
        self._apply_lora(target_modules, lora_rank, lora_alpha, lora_dropout)

        # Print stats
        total_lora_params = sum(l.num_params() for l in self.lora_layers.values())
        print(f"Applied LoRA to {len(self.lora_layers)} layers")
        print(f"Total LoRA parameters: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")

    def _apply_lora(
        self,
        target_modules: List[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        """Apply LoRA to target linear layers."""
        encoder = self.tokenizer.encoder

        for name, module in encoder.named_modules():
            if isinstance(module, nn.Linear):
                if any(target in name for target in target_modules):
                    # Create LoRA layer
                    lora_layer = LoRALayer(module, rank, alpha, dropout)
                    self.lora_layers[name] = lora_layer

                    # Replace the module
                    # Navigate to parent and replace
                    parts = name.split('.')
                    parent = encoder
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], lora_layer)

                    print(f"  Applied LoRA to: {name}")

        # Register LoRA layers as submodules for proper parameter tracking
        self.lora_modules = nn.ModuleDict({
            name.replace('.', '_'): layer for name, layer in self.lora_layers.items()
        })

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return only trainable (LoRA) parameters."""
        for layer in self.lora_layers.values():
            yield layer.lora_A
            yield layer.lora_B

    def num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(l.num_params() for l in self.lora_layers.values())

    def forward(self, *args, **kwargs):
        """Forward pass through the tokenizer (LoRA is integrated into layers)."""
        return self.tokenizer(*args, **kwargs)


class MiMoPartialFinetune(nn.Module):
    """
    MiMo encoder wrapper that only fine-tunes last N transformer layers.

    Earlier layers (general features) are frozen while later layers
    (more task-specific) are trainable.

    Args:
        tokenizer: MiMo tokenizer instance
        n_trainable_layers: Number of last layers to make trainable (default: 4)
    """

    def __init__(
        self,
        tokenizer,
        n_trainable_layers: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_trainable_layers = n_trainable_layers

        # First freeze all
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # Count and find transformer layers
        self.num_layers, self.transformer_layers = self._find_transformer_layers()
        print(f"MiMo encoder has {self.num_layers} transformer layers")

        # Unfreeze last N layers
        n = min(n_trainable_layers, self.num_layers)
        unfrozen_count = 0
        for i, layer in enumerate(self.transformer_layers):
            if i >= self.num_layers - n:
                for p in layer.parameters():
                    p.requires_grad = True
                unfrozen_count += 1

        print(f"Unfroze last {unfrozen_count} layers")
        print(f"Trainable parameters: {self.num_trainable_params():,} ({self.num_trainable_params()/1e6:.2f}M)")

    def _find_transformer_layers(self) -> tuple:
        """Find transformer layers in the encoder."""
        encoder = self.tokenizer.encoder
        layers = []

        for name, module in encoder.named_modules():
            if hasattr(module, 'layers') and isinstance(module.layers, nn.ModuleList):
                return len(module.layers), list(module.layers)
            if hasattr(module, 'blocks') and isinstance(module.blocks, nn.ModuleList):
                return len(module.blocks), list(module.blocks)

        # Fallback: collect individual layer modules
        layer_modules = {}
        for name, module in encoder.named_modules():
            for pattern in ['.layers.', '.blocks.', '.layer.']:
                if pattern in name:
                    parts = name.split(pattern)
                    if len(parts) > 1:
                        try:
                            idx = int(parts[1].split('.')[0])
                            if idx not in layer_modules:
                                layer_modules[idx] = module
                        except ValueError:
                            pass

        if layer_modules:
            max_idx = max(layer_modules.keys())
            layers = [layer_modules.get(i) for i in range(max_idx + 1) if i in layer_modules]
            return len(layers), layers

        return 0, []

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return only trainable parameters."""
        for p in self.tokenizer.parameters():
            if p.requires_grad:
                yield p

    def num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.tokenizer.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):
        """Forward pass through the tokenizer."""
        return self.tokenizer(*args, **kwargs)


class MiMoGradualUnfreeze(nn.Module):
    """
    Gradually unfreeze MiMo encoder layers during training.

    Starts with all layers frozen, then progressively unfreezes
    layers from the last (closest to output) to the first based
    on an epoch schedule.

    Args:
        tokenizer: MiMo tokenizer instance
        unfreeze_schedule: Dict mapping epoch -> number of layers to unfreeze
                          Example: {0: 0, 5: 4, 10: 8, 15: 16, 20: 32}
                          At epoch 0: all frozen
                          At epoch 5: last 4 layers unfrozen
                          etc.

    Example:
        >>> wrapper = MiMoGradualUnfreeze(tokenizer, {0: 0, 5: 4, 10: 8})
        >>> for epoch in range(20):
        ...     n_unfrozen = wrapper.on_epoch_start(epoch)
        ...     print(f"Epoch {epoch}: {n_unfrozen} layers unfrozen")
        ...     # training loop
    """

    def __init__(
        self,
        tokenizer,
        unfreeze_schedule: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.current_epoch = 0

        # Default schedule: gradual unfreezing over 20 epochs
        self.unfreeze_schedule = unfreeze_schedule or {
            0: 0,      # Start fully frozen
            5: 4,      # Unfreeze last 4 layers at epoch 5
            10: 8,     # Unfreeze last 8 layers at epoch 10
            15: 16,    # Unfreeze last 16 layers at epoch 15
            20: 32,    # Unfreeze all 32 layers at epoch 20
        }

        # Get transformer layers
        self.num_layers, self.transformer_layers = self._find_transformer_layers()
        print(f"MiMo encoder has {self.num_layers} transformer layers")
        print(f"Gradual unfreeze schedule: {self.unfreeze_schedule}")

        # Start fully frozen
        self._freeze_all()
        print(f"Starting with all {self.num_layers} layers frozen")

    def _find_transformer_layers(self) -> tuple:
        """Find transformer layers in the encoder."""
        encoder = self.tokenizer.encoder
        layers = []

        # Check common patterns
        for name, module in encoder.named_modules():
            if hasattr(module, 'layers') and isinstance(module.layers, nn.ModuleList):
                return len(module.layers), list(module.layers)
            if hasattr(module, 'blocks') and isinstance(module.blocks, nn.ModuleList):
                return len(module.blocks), list(module.blocks)

        # Fallback: collect layer modules by index
        layer_modules = {}
        for name, module in encoder.named_modules():
            for pattern in ['.layers.', '.blocks.', '.layer.']:
                if pattern in name:
                    parts = name.split(pattern)
                    if len(parts) > 1:
                        try:
                            idx = int(parts[1].split('.')[0])
                            if idx not in layer_modules:
                                layer_modules[idx] = module
                        except ValueError:
                            pass

        if layer_modules:
            max_idx = max(layer_modules.keys())
            layers = [layer_modules.get(i) for i in range(max_idx + 1) if i in layer_modules]
            return len(layers), layers

        return 32, []  # Default assumption for MiMo

    def _freeze_all(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.tokenizer.parameters():
            param.requires_grad = False

    def _unfreeze_last_n(self, n: int) -> None:
        """Unfreeze the last n transformer layers."""
        if n <= 0 or not self.transformer_layers:
            return

        n = min(n, self.num_layers)  # Cap at total layers

        # Unfreeze last n layers
        for layer in self.transformer_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def on_epoch_start(self, epoch: int) -> int:
        """
        Call at the start of each training epoch.
        Updates which layers are unfrozen based on schedule.

        Args:
            epoch: Current epoch number

        Returns:
            Number of layers currently unfrozen
        """
        self.current_epoch = epoch

        # Find the latest schedule point we've passed
        n_unfreeze = 0
        for schedule_epoch in sorted(self.unfreeze_schedule.keys()):
            if epoch >= schedule_epoch:
                n_unfreeze = self.unfreeze_schedule[schedule_epoch]

        # Reset to frozen, then unfreeze appropriate layers
        self._freeze_all()
        self._unfreeze_last_n(n_unfreeze)

        return n_unfreeze

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """Return only trainable parameters."""
        for p in self.tokenizer.parameters():
            if p.requires_grad:
                yield p

    def num_trainable_params(self) -> int:
        """Return count of currently trainable parameters."""
        return sum(p.numel() for p in self.tokenizer.parameters() if p.requires_grad)

    def get_frozen_status(self) -> Dict[str, str]:
        """Return dict showing frozen/unfrozen status of each layer."""
        status = {}
        for i, layer in enumerate(self.transformer_layers):
            is_frozen = not any(p.requires_grad for p in layer.parameters())
            status[f'layer_{i}'] = 'frozen' if is_frozen else 'unfrozen'
        return status

    def forward(self, *args, **kwargs):
        """Forward pass through the tokenizer."""
        return self.tokenizer(*args, **kwargs)


def get_layerwise_lr_params(
    tokenizer,
    base_lr: float = 1e-5,
    decay: float = 0.9,
    projection_module: Optional[nn.Module] = None,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Earlier layers get lower learning rates, later layers get higher rates.
    This allows fine-tuning all layers while maintaining stability.

    Args:
        tokenizer: MiMo tokenizer instance
        base_lr: Learning rate for the last layer
        decay: Decay factor per layer (earlier layers: base_lr * decay^depth)
        projection_module: Optional projection module to include at base_lr

    Returns:
        List of parameter groups for optimizer

    Example:
        params = get_layerwise_lr_params(tokenizer, base_lr=1e-5, decay=0.9)
        optimizer = AdamW(params)
    """
    encoder = tokenizer.encoder
    param_groups = []

    # Find transformer layers
    layers = []
    for name, module in encoder.named_modules():
        if hasattr(module, 'layers') and isinstance(module.layers, nn.ModuleList):
            layers = list(module.layers)
            break
        if hasattr(module, 'blocks') and isinstance(module.blocks, nn.ModuleList):
            layers = list(module.blocks)
            break

    num_layers = len(layers) if layers else 32

    if layers:
        # Add param groups for each layer with decaying LR
        for i, layer in enumerate(layers):
            # Earlier layers get lower LR
            layer_lr = base_lr * (decay ** (num_layers - i - 1))
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': layer_lr,
                'name': f'encoder_layer_{i}'
            })
            if i == 0 or i == num_layers - 1:
                print(f"Layer {i}: lr={layer_lr:.2e}")

    # Add non-layer encoder parameters at mid LR
    layer_params = set()
    for layer in layers:
        layer_params.update(id(p) for p in layer.parameters())

    other_encoder_params = [
        p for p in encoder.parameters()
        if id(p) not in layer_params
    ]
    if other_encoder_params:
        mid_lr = base_lr * (decay ** (num_layers // 2))
        param_groups.append({
            'params': other_encoder_params,
            'lr': mid_lr,
            'name': 'encoder_other'
        })

    # Add projection at base_lr (highest)
    if projection_module is not None:
        param_groups.append({
            'params': list(projection_module.parameters()),
            'lr': base_lr,
            'name': 'projection'
        })

    return param_groups


def apply_finetune_strategy(
    frontend,
    strategy: str = 'adapter',
    **kwargs
) -> nn.Module:
    """
    Factory function to apply a fine-tuning strategy to a MiMo frontend.

    Args:
        frontend: MiMoFrontend instance
        strategy: One of 'frozen', 'full', 'adapter', 'lora', 'partial', 'gradual'
        **kwargs: Strategy-specific arguments

    Returns:
        Modified frontend with fine-tuning strategy applied

    Strategy-specific kwargs:
        adapter:
            adapter_dim (int): Bottleneck dimension (default: 64)
            adapter_dropout (float): Dropout rate (default: 0.1)
            adapter_layers (str): 'all', 'last_n', 'every_n' (default: 'last_n')
            n_adapter_layers (int): Number of layers (default: 8)

        lora:
            lora_rank (int): Low-rank dimension (default: 8)
            lora_alpha (float): Scaling factor (default: 16)
            lora_dropout (float): Dropout rate (default: 0.0)
            target_modules (list): Module patterns to adapt

        partial:
            n_trainable_layers (int): Number of last layers to train (default: 4)

        gradual:
            schedule (dict): Epoch -> n_layers mapping (e.g., {0: 0, 10: 4, 20: 8})
            total_epochs (int): Used to auto-generate schedule if not provided

    Example:
        frontend = MiMoFrontend(...)
        frontend = apply_finetune_strategy(frontend, 'adapter', adapter_dim=64)
    """
    tokenizer = frontend.tokenizer

    if strategy == 'frozen':
        # Keep everything frozen (already default for MiMoFrontend with freeze=True)
        for p in tokenizer.parameters():
            p.requires_grad = False
        frontend._finetune_wrapper = None
        return frontend

    elif strategy == 'full':
        # Unfreeze everything
        for p in tokenizer.parameters():
            p.requires_grad = True
        frontend._finetune_wrapper = None
        total_params = sum(p.numel() for p in tokenizer.parameters())
        print(f"Full fine-tuning: {total_params:,} parameters ({total_params/1e9:.2f}B)")
        return frontend

    elif strategy == 'adapter':
        wrapper = MiMoWithAdapters(
            tokenizer,
            adapter_dim=kwargs.get('adapter_dim', 64),
            adapter_dropout=kwargs.get('adapter_dropout', 0.1),
            adapter_layers=kwargs.get('adapter_layers', 'last_n'),
            n_adapter_layers=kwargs.get('n_adapter_layers', 8),
        )
        frontend._finetune_wrapper = wrapper
        frontend._get_trainable_params = wrapper.get_trainable_params
        return frontend

    elif strategy == 'lora':
        wrapper = MiMoWithLoRA(
            tokenizer,
            lora_rank=kwargs.get('lora_rank', 8),
            lora_alpha=kwargs.get('lora_alpha', 16.0),
            lora_dropout=kwargs.get('lora_dropout', 0.0),
            target_modules=kwargs.get('target_modules', None),
        )
        frontend._finetune_wrapper = wrapper
        frontend._get_trainable_params = wrapper.get_trainable_params
        return frontend

    elif strategy == 'partial':
        wrapper = MiMoPartialFinetune(
            tokenizer,
            n_trainable_layers=kwargs.get('n_trainable_layers', 4),
        )
        frontend._finetune_wrapper = wrapper
        frontend._get_trainable_params = wrapper.get_trainable_params
        return frontend

    elif strategy == 'gradual':
        # Parse schedule from kwargs or use default
        schedule = kwargs.get('schedule', None)
        if schedule is None:
            # Default schedule based on total epochs
            total_epochs = kwargs.get('total_epochs', 100)
            # Unfreeze in 5 stages over training
            step = total_epochs // 5
            schedule = {
                0: 0,
                step: 4,
                step * 2: 8,
                step * 3: 16,
                step * 4: 32,
            }

        wrapper = MiMoGradualUnfreeze(
            tokenizer,
            unfreeze_schedule=schedule,
        )
        frontend._finetune_wrapper = wrapper
        frontend._get_trainable_params = wrapper.get_trainable_params
        return frontend

    else:
        raise ValueError(
            f"Unknown finetune strategy: {strategy}. "
            "Choose from: frozen, full, adapter, lora, partial, gradual"
        )
