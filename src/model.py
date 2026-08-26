import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CNN_CHECKPOINT_ABOVE = 256

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)


def _bn_safe(fn, bns):
    state = {"first": True}

    def wrapped(x):
        if state["first"]:
            state["first"] = False
            return fn(x)
        saved = [(b.momentum, None if b.num_batches_tracked is None
                  else b.num_batches_tracked.clone()) for b in bns]
        for b in bns:
            b.momentum = 0.0
        try:
            return fn(x)
        finally:
            for b, (momentum, nbt) in zip(bns, saved):
                b.momentum = momentum
                if nbt is not None:
                    b.num_batches_tracked.copy_(nbt)

    return wrapped


class _CheckpointedBranch(nn.Module):
    def _run(self, x):
        raise NotImplementedError

    def forward(self, x):
        if (self.checkpoint_above and self.training and torch.is_grad_enabled()
                and x.shape[0] > self.checkpoint_above):
            # Collect BatchNorms recursively (self.modules()) rather than from a
            # flat self.conv - the ResNet branch nests them inside residual blocks,
            # and for the plain branches this yields the exact same set.
            bns = [m for m in self.modules() if isinstance(m, _BN_TYPES)]
            return checkpoint(_bn_safe(self._run, bns), x, use_reentrant=False)
        return self._run(x)


class CNNBranch(_CheckpointedBranch):
    def __init__(self, in_channels=1, conv_channels=(16, 32), kernel_size=7, pool_size=4,
                 checkpoint_above=CNN_CHECKPOINT_ABOVE):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool_size),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.out_dim = conv_channels[-1]
        self.checkpoint_above = checkpoint_above

    def _run(self, x):
        # x: (N, in_channels, window_size)
        return self.conv(x).mean(dim=-1)


class CNN2DBranch(_CheckpointedBranch):
    def __init__(self, in_channels=1, conv_channels=(8, 16, 32), kernel_size=3, pool_size=2,
                 checkpoint_above=CNN_CHECKPOINT_ABOVE):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.out_dim = conv_channels[-1]
        self.checkpoint_above = checkpoint_above

    def _run(self, x):
        # x: (N, in_channels, F, T)
        return self.conv(x).mean(dim=(-2, -1))


ANN_POOL_LEN = 256


class ANNBranch(_CheckpointedBranch):
    """Plain fully-connected (MLP) per-window feature extractor for 1D inputs - a
    drop-in alternative to CNNBranch, i.e. the "no convolution at all" baseline.

    A window is first adaptive-average-pooled along time to a fixed `pool_len`
    (default 256) and then flattened, so the first Linear sees
    in_channels*pool_len features regardless of the raw window length. That
    pooling is what makes this usable here: the raw windows can be tens of
    thousands of samples (5 s at 10 kHz), and flattening one of those straight
    into a Linear would be a multi-million-parameter first layer per branch.
    Pooling is an average, not a stride/decimation, so it is an anti-aliased
    summary rather than a subsample.

    `hidden` gives the MLP widths (Linear->BN->ReLU->Dropout per entry), reusing
    the per-branch *_channels args so out_dim == hidden[-1] and the fused LSTM
    input width is identical to the CNN branches'. Same in/out contract as
    CNNBranch, so it slots into _apply_branch and the checkpointing path
    unchanged."""
    def __init__(self, in_channels=1, hidden=(16, 32, 64), pool_len=ANN_POOL_LEN,
                 dropout=0.0, checkpoint_above=CNN_CHECKPOINT_ABOVE):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(pool_len)
        layers = []
        d_in = in_channels * pool_len
        for d_out in hidden:
            layers += [
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = d_out
        self.mlp = nn.Sequential(*layers)
        self.out_dim = hidden[-1]
        self.checkpoint_above = checkpoint_above

    def _run(self, x):
        # x: (N, in_channels, window_size) -> (N, out_dim)
        x = self.pool(x)
        return self.mlp(x.flatten(1))


def _expand_blocks(blocks_per_stage, n_stages):
    """`blocks_per_stage` may be an int (same count for every stage) or a
    per-stage sequence; returns a length-`n_stages` list. Standard ResNet-18 is
    2 blocks per stage over 4 stages, i.e. (2, 2, 2, 2)."""
    if isinstance(blocks_per_stage, int):
        return [blocks_per_stage] * n_stages
    bps = list(blocks_per_stage)
    if len(bps) != n_stages:
        raise ValueError(f"blocks_per_stage has {len(bps)} entries but there are "
                         f"{n_stages} stages (len(conv_channels)={n_stages}).")
    return bps


class _BasicBlock1D(nn.Module):
    """One residual block for the 1D ResNet branch: two Conv1d->BN->ReLU with a
    skip connection. When the block changes width or strides (downsamples), the
    skip is projected by a 1x1 conv so the shapes still add. Conv bias is off
    because the following BatchNorm already supplies a shift."""
    def __init__(self, c_in, c_out, stride=1, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, stride=1, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet1DBranch(_CheckpointedBranch):
    """1D ResNet per-window feature extractor - a drop-in alternative to CNNBranch.

    A strided stem (conv stride 2 + maxpool stride 2) downsamples the raw window
    ~4x immediately, then one residual stage per entry in `conv_channels` (the
    first stage keeps stride 1; later stages downsample 2x while doubling width),
    then global average pooling -> (N, conv_channels[-1]). Versus the plain
    stacked-conv CNNBranch the point is the residual skips, which keep depth
    trainable. Activation memory is roughly comparable - actually a little HIGHER
    than CNNBranch (the extra residual depth retains more intermediates for
    backward, which outweighs the striding savings), but it stays bounded by the
    same gradient-checkpointing path. Same in/out contract as CNNBranch - it slots
    into _apply_branch and the checkpointing path unchanged, and out_dim equals
    conv_channels[-1] so the fused LSTM input width is identical."""
    def __init__(self, in_channels=1, conv_channels=(16, 32, 64), kernel_size=7,
                 blocks_per_stage=1, checkpoint_above=CNN_CHECKPOINT_ABOVE):
        super().__init__()
        c0 = conv_channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c0, kernel_size=kernel_size, stride=2,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        c_in = c0
        bps = _expand_blocks(blocks_per_stage, len(conv_channels))
        for i, c_out in enumerate(conv_channels):
            stage_stride = 1 if i == 0 else 2   # stem already downsampled ~4x
            for b in range(bps[i]):
                blocks.append(_BasicBlock1D(c_in, c_out,
                                            stride=stage_stride if b == 0 else 1))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = conv_channels[-1]
        self.checkpoint_above = checkpoint_above

    def _run(self, x):
        # x: (N, in_channels, window_size) -> (N, out_dim)
        x = self.stem(x)
        x = self.blocks(x)
        return x.mean(dim=-1)


class _BasicBlock2D(nn.Module):
    """2D analogue of _BasicBlock1D, for the CWT scalogram ResNet branch: two
    Conv2d->BN->ReLU with a skip that's 1x1-projected when width or stride
    changes. Conv bias off because BatchNorm supplies the shift."""
    def __init__(self, c_in, c_out, stride=1, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size, stride=1, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CNN2DResNetBranch(_CheckpointedBranch):
    """2D ResNet over each window's CWT scalogram - a drop-in alternative to
    CNN2DBranch. Strided stem (conv s2 + maxpool s2) then one residual stage per
    entry in `conv_channels` (first stride 1, later stride 2 while doubling
    width), then global average pool over (freq, time) -> (N, conv_channels[-1]).
    Same in/out contract and out_dim as CNN2DBranch, so it slots into
    _apply_branch_2d and the fused LSTM width unchanged. This is the branch where
    image-style residual nets genuinely apply - the scalogram IS a 2D image, so
    (unlike the 1D signals) a 2D ResNet/Inception-ResNet is the natural fit here."""
    def __init__(self, in_channels=1, conv_channels=(8, 16, 32), kernel_size=7,
                 blocks_per_stage=1, checkpoint_above=CNN_CHECKPOINT_ABOVE):
        super().__init__()
        c0 = conv_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=kernel_size, stride=2,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        c_in = c0
        bps = _expand_blocks(blocks_per_stage, len(conv_channels))
        for i, c_out in enumerate(conv_channels):
            stage_stride = 1 if i == 0 else 2   # stem already downsampled ~4x
            for b in range(bps[i]):
                blocks.append(_BasicBlock2D(c_in, c_out,
                                            stride=stage_stride if b == 0 else 1))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = conv_channels[-1]
        self.checkpoint_above = checkpoint_above

    def _run(self, x):
        # x: (N, in_channels, F, T) -> (N, out_dim)
        x = self.stem(x)
        x = self.blocks(x)
        return x.mean(dim=(-2, -1))


class AttentionPool(nn.Module):
    def __init__(self, in_dim, attn_dim=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, seq, return_weights=False):
        # seq: (B, T, D) -> context: (B, D)
        scores = self.score(seq)                 # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # over time
        context = (weights * seq).sum(dim=1)     # (B, D)
        if return_weights:
            return context, weights.squeeze(-1)  # (B, D), (B, T)
        return context


class SKNABPModel(nn.Module):
    def __init__(self, ecg_channels=(16, 32, 64), skna_channels=(16, 32, 64),
                 iskna_rskna_channels=(16, 32, 64), kernel_size=7, pool_size=4,
                 lstm_hidden=64, lstm_layers=2, dropout=0.3,
                 use_ecg=True, use_skna=True, use_iskna_rskna=True, use_askna=True,
                 use_fd=False, fd_channels=(16, 32, 64),
                 use_fft=False, fft_channels=(16, 32, 64),
                 use_psd=False, psd_channels=(16, 32, 64),
                 use_cwt=False, cwt_channels=(8, 16, 32),
                 cnn_checkpoint_above=CNN_CHECKPOINT_ABOVE, cnn_arch="plain",
                 resnet_blocks_per_stage=1, cwt_arch=None, ann_pool_len=ANN_POOL_LEN):
        super().__init__()
        ckpt = cnn_checkpoint_above
        if cnn_arch not in ("plain", "resnet", "resnet18", "ann"):
            raise ValueError("cnn_arch must be 'plain', 'resnet', 'resnet18', or 'ann', "
                             f"got {cnn_arch!r}.")
        # The 2D (CWT scalogram) branch has its own architecture selector so the
        # 1D and 2D halves can be chosen independently - e.g. an ANN over the
        # waveforms with a 2D ResNet-18 over the scalogram. None = "follow
        # cnn_arch", which is what every pre-cwt_arch checkpoint did; "ann" is 1D
        # only, so under it a CWT branch with no explicit choice falls back to the
        # plain 2D CNN.
        if cwt_arch is None:
            cwt_arch = "plain" if cnn_arch == "ann" else cnn_arch
        if cwt_arch not in ("plain", "resnet", "resnet18"):
            raise ValueError(f"cwt_arch must be 'plain', 'resnet', or 'resnet18', got {cwt_arch!r}.")
        self.cnn_arch = cnn_arch
        self.cwt_arch = cwt_arch
        self.resnet_blocks_per_stage = resnet_blocks_per_stage
        self.ann_pool_len = ann_pool_len
        # Every input branch is individually switchable so the CNN's inputs can be
        # ablated (e.g. ECG-only or SKNA-only). A disabled branch is not built and
        # contributes nothing to the fused feature vector fed to the LSTM.
        self.use_ecg = use_ecg
        self.use_skna = use_skna
        self.use_iskna_rskna = use_iskna_rskna
        self.use_askna = use_askna
        # FD is the 2-channel (FFT, PSD) spectrum. `use_fd` is the legacy single
        # 2-channel branch (kept so old checkpoints still reconstruct); `use_fft`
        # and `use_psd` are the split form - one independent 1-channel branch each,
        # both fed by channel-slicing the same fd_signal.npz (ch0=FFT, ch1=PSD).
        self.use_fd = use_fd
        self.use_fft = use_fft
        self.use_psd = use_psd
        self.use_cwt = use_cwt

        # cnn_arch selects the family used for every 1D branch:
        #   "plain"    -> the original stacked-conv CNNBranch.
        #   "resnet"   -> a residual 1D ResNet with `resnet_blocks_per_stage` blocks
        #                 per stage and the widths taken from the *_channels args.
        #   "resnet18" -> the canonical ResNet-18 topology: 4 stages of [2,2,2,2]
        #                 BasicBlocks at the standard widths (64,128,256,512).
        #                 Because those widths ARE part of the ResNet-18 definition,
        #                 this ignores the per-branch *_channels args.
        #   "ann"      -> no convolution at all: adaptive-avg-pool the window to
        #                 `ann_pool_len` samples, flatten, and run an MLP whose
        #                 hidden widths are the *_channels args. This is the plain
        #                 fully-connected baseline, so it applies to EVERY 1D input
        #                 including the spectral ones (FD / FFT / PSD) - under the
        #                 conv archs those stay plain CNNs, since a coarse spectrum
        #                 is neither a waveform nor an image.
        # cwt_arch independently selects the family for the 2D scalogram branch
        # ("plain" 2D CNN, 2D "resnet", or a 2D "resnet18"), so e.g. ANN-on-1D +
        # ResNet-18-on-CWT is expressible. In every case the in/out contract is
        # identical (each branch reduces to out_dim = its last width), so the fused
        # LSTM input width just adapts.
        is_resnet = cnn_arch in ("resnet", "resnet18")
        is_ann = cnn_arch == "ann"
        R18_CHANNELS = (64, 128, 256, 512)
        res_blocks = 2 if cnn_arch == "resnet18" else resnet_blocks_per_stage
        cwt_res_blocks = 2 if cwt_arch == "resnet18" else resnet_blocks_per_stage

        def _time_branch(in_ch, channels):
            if is_ann:
                return ANNBranch(in_ch, channels, pool_len=ann_pool_len,
                                 dropout=dropout, checkpoint_above=ckpt)
            if is_resnet:
                ch = R18_CHANNELS if cnn_arch == "resnet18" else channels
                return ResNet1DBranch(in_ch, ch, kernel_size=kernel_size,
                                      blocks_per_stage=res_blocks, checkpoint_above=ckpt)
            return CNNBranch(in_ch, channels, kernel_size, pool_size, ckpt)

        def _spectral_branch(in_ch, channels):
            # Coarse spectra (FD / FFT / PSD): plain 1D CNN under every conv arch,
            # but an MLP under "ann" so that option really is convolution-free.
            if is_ann:
                return ANNBranch(in_ch, channels, pool_len=ann_pool_len,
                                 dropout=dropout, checkpoint_above=ckpt)
            return CNNBranch(in_ch, channels, kernel_size, pool_size, ckpt)

        def _cwt_branch(channels):
            if cwt_arch in ("resnet", "resnet18"):
                ch = R18_CHANNELS if cwt_arch == "resnet18" else channels
                return CNN2DResNetBranch(1, ch, blocks_per_stage=cwt_res_blocks,
                                         checkpoint_above=ckpt)
            return CNN2DBranch(1, channels, checkpoint_above=ckpt)

        self.ecg_branch = _time_branch(1, ecg_channels) if use_ecg else None
        self.skna_branch = _time_branch(1, skna_channels) if use_skna else None
        self.iskna_rskna_branch = _time_branch(2, iskna_rskna_channels) if use_iskna_rskna else None
        self.fd_branch = _spectral_branch(2, fd_channels) if use_fd else None
        # FFT/PSD each get their own independent 1-channel branch (coarse spectra,
        # not waveforms), sliced out of the 2-channel fd input in forward().
        self.fft_branch = _spectral_branch(1, fft_channels) if use_fft else None
        self.psd_branch = _spectral_branch(1, psd_channels) if use_psd else None
        self.cwt_branch = _cwt_branch(cwt_channels) if use_cwt else None

        feat_dim = ((self.ecg_branch.out_dim if use_ecg else 0)
                    + (self.skna_branch.out_dim if use_skna else 0)
                    + (self.iskna_rskna_branch.out_dim if use_iskna_rskna else 0)
                    + (1 if use_askna else 0)
                    + (self.fd_branch.out_dim if use_fd else 0)
                    + (self.fft_branch.out_dim if use_fft else 0)
                    + (self.psd_branch.out_dim if use_psd else 0)
                    + (self.cwt_branch.out_dim if use_cwt else 0))
        if feat_dim == 0:
            raise ValueError("No CNN inputs enabled - enable at least one of "
                             "ecg / skna / iskna_rskna / askna / fd / fft / psd / cwt.")

        self.lstm = nn.LSTM(
            input_size=feat_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_pool = AttentionPool(lstm_hidden * 2, attn_dim=lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 2, 2)  # -> [SBP, DBP]

    def _apply_branch(self, branch, x):
        B, T, C, L = x.shape
        return branch(x.reshape(B * T, C, L)).reshape(B, T, -1)

    def _apply_branch_2d(self, branch, x):
        B, T, C, Fdim, W = x.shape
        return branch(x.reshape(B * T, C, Fdim, W)).reshape(B, T, -1)

    def forward(self, ecg, skna, rskna, iskna, askna=None, fd=None, cwt=None, return_attn=False):
        feats = []
        if self.use_ecg:
            feats.append(self._apply_branch(self.ecg_branch, ecg))
        if self.use_skna:
            feats.append(self._apply_branch(self.skna_branch, skna))
        if self.use_iskna_rskna:
            iskna_rskna = torch.cat([rskna, iskna], dim=2)   # (B, T, 2, L)
            feats.append(self._apply_branch(self.iskna_rskna_branch, iskna_rskna))
        if self.use_askna:
            if askna is None:
                raise ValueError("use_askna=True but no askna tensor was passed to forward().")
            feats.append(askna)
        if self.use_fd:
            if fd is None:
                raise ValueError("use_fd=True but no fd tensor was passed to forward().")
            feats.append(self._apply_branch(self.fd_branch, fd))
        if self.use_fft or self.use_psd:
            if fd is None:
                raise ValueError("use_fft/use_psd=True but no fd tensor was passed to forward().")
            # fd is (B, T, 2, L): channel 0 = FFT magnitude, channel 1 = PSD.
            if self.use_fft:
                feats.append(self._apply_branch(self.fft_branch, fd[:, :, 0:1, :]))
            if self.use_psd:
                feats.append(self._apply_branch(self.psd_branch, fd[:, :, 1:2, :]))
        if self.use_cwt:
            if cwt is None:
                raise ValueError("use_cwt=True but no cwt tensor was passed to forward().")
            feats.append(self._apply_branch_2d(self.cwt_branch, cwt))
        x = torch.cat(feats, dim=-1)     

        seq_out, _ = self.lstm(x)        
        if return_attn:
            h, attn_weights = self.attn_pool(seq_out, return_weights=True) 
            h = self.dropout(h)
            return self.head(h), attn_weights
        h = self.attn_pool(seq_out)       
        h = self.dropout(h)
        return self.head(h)             


class TargetScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    def fit(self, targets):
        # targets: (N, 2) tensor of [sbp, dbp]
        self.mean = targets.mean(dim=0)
        std = targets.std(dim=0)
        self.std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return self
    def transform(self, targets):
        return (targets - self.mean.to(targets.device)) / self.std.to(targets.device)
    def inverse_transform(self, targets):
        return targets * self.std.to(targets.device) + self.mean.to(targets.device)
    def state_dict(self):
        return {"mean": self.mean, "std": self.std}
    @classmethod
    def from_state_dict(cls, sd):
        return cls(mean=sd["mean"], std=sd["std"])


def freeze_for_hybrid_calibration(model):
    """
    Freeze every parameter except the LAST LSTM layer (both directions) and
    the attention-pool + final FC head, mirroring the "hybrid calibration"
    fine-tune in Xiang et al. 2025 (McBP-Net): after population pretraining,
    only the last LSTM module and the fully-connected layer are updated on a
    per-subject calibration slice - the CNN feature extractors and earlier
    LSTM layers stay at their population-trained weights, so the fine-tune
    can only re-map already-extracted features to this one subject's BP
    scale/offset, not re-learn features from a handful of calibration windows.
    """
    for prm in model.parameters():
        prm.requires_grad = False
    last_layer = model.lstm.num_layers - 1
    for name, prm in model.lstm.named_parameters():
        # names look like 'weight_ih_l1' or 'weight_hh_l1_reverse'
        layer_idx = int(name.split("_l", 1)[1].split("_", 1)[0])
        if layer_idx == last_layer:
            prm.requires_grad = True
    for prm in model.attn_pool.parameters():
        prm.requires_grad = True
    for prm in model.head.parameters():
        prm.requires_grad = True


def unfreeze_all(model):
    for prm in model.parameters():
        prm.requires_grad = True


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False
    def step(self, value):
        """Returns True if `value` is a new best."""
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False

def batch_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def run_batch(model, batch, device, target_scaler=None):
    batch = batch_to_device(batch, device)
    pred = model(batch["ecg"], batch["skna"], batch["rskna"], batch["iskna"],
                 batch.get("askna"), batch.get("fd"), batch.get("cwt"))
    target_abs = torch.stack([batch["sbp"], batch["dbp"]], dim=-1).to(device) 
    baseline = torch.stack([batch["baseline_sbp"], batch["baseline_dbp"]], dim=-1).to(device)  
    target_delta = target_abs - baseline
    target_delta_norm = target_scaler.transform(target_delta) if target_scaler is not None else target_delta
    return pred, target_delta_norm, target_delta, target_abs, baseline


def bp_loss(pred, target, anti_collapse_lambda=0.0):
    base = F.smooth_l1_loss(pred, target)
    if anti_collapse_lambda > 0 and pred.shape[0] > 1:
        pred_std = pred.std(dim=0, unbiased=False)
        target_std = target.std(dim=0, unbiased=False)
        collapse_penalty = F.relu(target_std - pred_std).mean()
        return base + anti_collapse_lambda * collapse_penalty
    return base


def train_one_epoch(model, loader, optimizer, device, target_scaler=None, anti_collapse_lambda=0.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        pred, target_delta_norm, _, _, _ = run_batch(model, batch, device, target_scaler)
        loss = bp_loss(pred, target_delta_norm, anti_collapse_lambda)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.shape[0]
        n += pred.shape[0]
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, target_scaler=None, return_predictions=False, anti_collapse_lambda=0.0):
    model.eval()
    total_loss = 0.0
    abs_err_sbp, abs_err_dbp = 0.0, 0.0
    delta_err_sbp, delta_err_dbp = 0.0, 0.0
    n = 0
    all_pred_mmhg, all_true_mmhg, all_subjects = [], [], []
    all_pred_delta, all_true_delta = [], []

    for batch in loader:
        pred, target_delta_norm, target_delta, target_abs, baseline = run_batch(model, batch, device, target_scaler)
        loss = bp_loss(pred, target_delta_norm, anti_collapse_lambda)
        total_loss += loss.item() * pred.shape[0]

        pred_delta_mmhg = target_scaler.inverse_transform(pred) if target_scaler is not None else pred
        pred_abs_mmhg = baseline + pred_delta_mmhg

        err_abs = (pred_abs_mmhg - target_abs).abs()
        abs_err_sbp += err_abs[:, 0].sum().item()
        abs_err_dbp += err_abs[:, 1].sum().item()

        err_delta = (pred_delta_mmhg - target_delta).abs()
        delta_err_sbp += err_delta[:, 0].sum().item()
        delta_err_dbp += err_delta[:, 1].sum().item()
        n += pred.shape[0]

        if return_predictions:
            all_pred_mmhg.append(pred_abs_mmhg.cpu())
            all_true_mmhg.append(target_abs.cpu())
            all_pred_delta.append(pred_delta_mmhg.cpu())
            all_true_delta.append(target_delta.cpu())
            all_subjects.extend(batch["subject_id"])

    result = {
        "loss": total_loss / n,
        "MAE_SBP": abs_err_sbp / n, "MAE_DBP": abs_err_dbp / n,
        "MAE_SBP_delta": delta_err_sbp / n, "MAE_DBP_delta": delta_err_dbp / n,
    }
    if return_predictions:
        pred_arr = torch.cat(all_pred_mmhg, dim=0).numpy()
        true_arr = torch.cat(all_true_mmhg, dim=0).numpy()
        pred_delta_arr = torch.cat(all_pred_delta, dim=0).numpy()
        true_delta_arr = torch.cat(all_true_delta, dim=0).numpy()
        result["pred"] = pred_arr
        result["true"] = true_arr
        result["pred_delta"] = pred_delta_arr
        result["true_delta"] = true_delta_arr
        result["subject_id"] = all_subjects

        def safe_corr(a, b):
            if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
                return float('nan')
            return float(np.corrcoef(a, b)[0, 1])

        result["corr_SBP"] = safe_corr(pred_arr[:, 0], true_arr[:, 0])
        result["corr_DBP"] = safe_corr(pred_arr[:, 1], true_arr[:, 1])
        result["corr_SBP_delta"] = safe_corr(pred_delta_arr[:, 0], true_delta_arr[:, 0])
        result["corr_DBP_delta"] = safe_corr(pred_delta_arr[:, 1], true_delta_arr[:, 1])

        result["pred_std_SBP"] = float(np.std(pred_arr[:, 0]))
        result["pred_std_DBP"] = float(np.std(pred_arr[:, 1]))
        result["true_std_SBP"] = float(np.std(true_arr[:, 0]))
        result["true_std_DBP"] = float(np.std(true_arr[:, 1]))

        # ME (mean error) and SD of the error - the AAMI/ISO device-validation
        # pair, reported as "ME +/- SD". ME == bias == mean(pred - true); SD is
        # the spread of the signed error, ddof=1 to match bp_standards.py.
        err_sbp = pred_arr[:, 0] - true_arr[:, 0]
        err_dbp = pred_arr[:, 1] - true_arr[:, 1]
        result["bias_SBP"] = float(np.mean(err_sbp))
        result["bias_DBP"] = float(np.mean(err_dbp))
        result["SDE_SBP"] = float(np.std(err_sbp, ddof=1)) if len(err_sbp) > 1 else float("nan")
        result["SDE_DBP"] = float(np.std(err_dbp, ddof=1)) if len(err_dbp) > 1 else float("nan")
        result["MAE_SBP_debiased"] = float(np.mean(np.abs(
            true_arr[:, 0] - (pred_arr[:, 0] - result["bias_SBP"]))))
        result["MAE_DBP_debiased"] = float(np.mean(np.abs(
            true_arr[:, 1] - (pred_arr[:, 1] - result["bias_DBP"]))))

        subj_arr = np.asarray(all_subjects)
        oracle_sbp = np.empty(len(true_arr), dtype=np.float64)
        oracle_dbp = np.empty(len(true_arr), dtype=np.float64)
        for s in np.unique(subj_arr):
            m = subj_arr == s
            oracle_sbp[m] = true_arr[m, 0].mean()
            oracle_dbp[m] = true_arr[m, 1].mean()
        result["MAE_SBP_oracle_const"] = float(np.mean(np.abs(true_arr[:, 0] - oracle_sbp)))
        result["MAE_DBP_oracle_const"] = float(np.mean(np.abs(true_arr[:, 1] - oracle_dbp)))
        result["has_tracking_skill_SBP"] = bool(
            result["MAE_SBP_debiased"] < result["MAE_SBP_oracle_const"])
        result["has_tracking_skill_DBP"] = bool(
            result["MAE_DBP_debiased"] < result["MAE_DBP_oracle_const"])
    return result