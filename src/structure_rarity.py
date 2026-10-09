"""
structure_rarity.py
The proposed loss for RQ3 (Methodology section 38): two additions to the
existing contrastive losses of train_contrastive.py.

1. Same-group negatives (structure). Every training package belongs to a
   connectivity group: connected (it has at least one neighbour in the graph)
   or isolated. Negatives are drawn only from the anchor's own group, so the
   encoder cannot separate packages by how connected they are (CoLA's failure,
   section 24). DGI: one summary per group.

2. Rarity-shifted negatives (content). A shifted copy of a package is the same
   package, in the same sampled neighbourhood, with one or a few RARE features
   changed. Rarity is measured on the TRAINING packages only, without labels:
   variability of a feature = share of training packages not at its most
   common value; rare = 0 < variability <= nu. A feature at its common value
   gets a value drawn from the training packages' uncommon values (a 0/1 flag
   is flipped); a feature away from it is set back to the common value. The
   copy is an extra, structure-matched hard negative, which keeps the encoder
   sensitive to the rare features it otherwise suppresses (section 36).
   "uniform" chooses among ALL varying features instead (ablation N4: does
   rarity matter, or would any shift do?).

Ablation conditions (section 3.7.3)
  N1  same-group negatives only          --strata conn
  N2  shifted negatives only             --shift rare
  N3  both (full)                        --strata conn --shift rare
  N4  both, shift chosen uniformly       --strata conn --shift uniform

Per loss
  InfoNCE  shifted copies enter the denominator with weight lambda (--shift-weight):
             l_i = -log  e^{s(i,i+)/t} / ( e^{s(i,i+)/t} + sum_{j != i, g(j) = g(i)} e^{s(i,j)/t}
                                           + lambda sum_k e^{s(i, i~k)/t} )
  Triplet  hinge with a same-group random (or hardest same-group) negative,
           plus beta (--shift-weight) x hinge with the shifted copy
  DGI      corruption = shifted copies (structure kept) instead of row
           shuffling; summary per group. --shift-weight is not used.

Nothing here reads labels, splits or families.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def connectivity_groups(graph, nodes):
    """1 = connected (at least one neighbour in the graph the encoder samples from), 0 = isolated."""
    return (graph.degree(np.asarray(nodes)) > 0).astype(np.int64)


# ---------------------------------------------------------------------------
# Shifted copies
# ---------------------------------------------------------------------------
class RarityShifter:
    """Shift sampler fitted on the training packages' feature rows (as the encoder sees them).

    x_train   [N, F] tensor, own feature rows of the training packages
    mode      "rare" (0 < variability <= nu) or "uniform" (any varying feature)
    """

    def __init__(self, x_train, mode="rare", nu=0.05, names=None, atol=1e-6):
        if mode not in ("rare", "uniform"):
            raise ValueError(mode)
        X = x_train.detach().cpu().numpy().astype(np.float64)
        n, f = X.shape
        self.atol = atol
        self.mode_val = np.zeros(f)
        self.pools = [np.zeros(0)] * f
        self.variability = np.zeros(f)
        for j in range(f):
            vals, counts = np.unique(np.round(X[:, j], 6), return_counts=True)
            m = vals[np.argmax(counts)]
            off = ~np.isclose(X[:, j], m, atol=atol)
            self.mode_val[j] = X[~off, j].mean() if (~off).any() else m   # exact stored value, not the rounded one
            self.pools[j] = X[off, j]
            self.variability[j] = off.mean()
        varying = self.variability > 0
        self.candidates = np.where(varying & (self.variability <= nu))[0] if mode == "rare" else np.where(varying)[0]
        if len(self.candidates) == 0:
            raise ValueError(f"no candidate features for mode {mode} (nu {nu})")
        self.mode, self.nu = mode, nu
        self.names = list(names) if names is not None else [str(j) for j in range(f)]

    def describe(self, k=12):
        order = self.candidates[np.argsort(self.variability[self.candidates])]
        shown = ", ".join(f"{self.names[j]} ({self.variability[j]:.1%})" for j in order[:k])
        more = f", ... (+{len(order) - k})" if len(order) > k else ""
        return f"{len(self.candidates)} {self.mode} features (nu {self.nu:.0%}): {shown}{more}"

    def shift(self, X, n_feat, rng, allowed=None):
        """Shifted copies of the rows of X [B, F] (tensor). allowed [F] bool: columns that may
        be changed (e.g. not masked in the view the copy is built from). Returns (X_shifted, cols [B, k])."""
        cand = self.candidates
        if allowed is not None:
            allowed = np.asarray(allowed, dtype=bool)
            cand = cand[allowed[cand]]
        if len(cand) == 0:            # every candidate masked in this view: no shift possible
            return X.clone(), np.zeros((X.shape[0], 0), dtype=np.int64)
        B = X.shape[0]
        k = min(n_feat, len(cand))
        cols = cand[np.argsort(rng.random((B, len(cand))), axis=1)[:, :k]]   # k distinct columns per row
        Xn = X.detach().cpu().numpy().astype(np.float64).copy()
        rows = np.repeat(np.arange(B), k)
        flat = cols.ravel()
        cur = Xn[rows, flat]
        at_mode = np.isclose(cur, self.mode_val[flat], atol=self.atol)
        new = self.mode_val[flat].copy()                     # away from the common value -> back to it
        for j in np.unique(flat[at_mode]):                   # at the common value -> an uncommon training value
            sel = at_mode & (flat == j)
            new[sel] = self.pools[j][rng.integers(0, len(self.pools[j]), size=int(sel.sum()))]
        Xn[rows, flat] = new
        return torch.from_numpy(Xn).to(X.dtype), cols


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def infonce_sr(z1, z2, tau, groups=None, z_shift=None, weight=1.0):
    """NT-Xent with optional same-group negatives and shifted negatives.
    z1, z2 [n, d] projections of the two views; groups [n] (numpy/tensor) or None;
    z_shift [K, n, d] projections of shifted copies or None. With groups=None and
    z_shift=None this is exactly train_contrastive.infonce."""
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    n = len(z1)
    sim.fill_diagonal_(float("-inf"))
    if groups is not None:
        g = torch.as_tensor(np.asarray(groups), device=z.device)
        g2 = torch.cat([g, g])
        sim = sim.masked_fill(g2[:, None] != g2[None, :], float("-inf"))   # positives share the group
    if z_shift is not None:
        zs = F.normalize(z_shift, dim=-1)                                 # [K, n, d]
        extra = torch.cat([torch.einsum("nd,knd->nk", z1, zs), torch.einsum("nd,knd->nk", z2, zs)]) / tau
        sim = torch.cat([sim, extra + math.log(weight)], dim=1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def same_group_partner(groups, rng):
    """For every i a random j != i from the same group (any other package if i is alone in its group)."""
    groups = np.asarray(groups)
    n = len(groups)
    out = np.empty(n, dtype=np.int64)
    for g in np.unique(groups):
        members = np.where(groups == g)[0]
        if len(members) > 1:
            out[members] = members[(np.arange(len(members)) + int(rng.integers(1, len(members)))) % len(members)]
        else:
            others = np.where(np.arange(n) != members[0])[0]
            out[members] = others[rng.integers(0, len(others))] if len(others) else members
    return out


def triplet_sr(z1, z2, margin, mining, rng, groups=None, z_shift=None, weight=1.0):
    """Triplet loss (L2 on normalised projections) with optional same-group negatives and
    a second hinge for shifted copies. With groups=None and z_shift=None this is exactly
    train_contrastive.triplet."""
    a, p = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    n = len(a)
    d_pos = (a - p).norm(dim=-1)
    if mining == "hard":
        d = torch.cdist(a, p)
        d = d + torch.eye(n, device=a.device) * 1e6
        if groups is not None:
            g = torch.as_tensor(np.asarray(groups), device=a.device)
            other = g[:, None] != g[None, :]
            alone = (~other).sum(1) <= 1                                     # only itself in its group
            d = d + (other & ~alone[:, None]).float() * 1e6
        d_neg = d.min(dim=1).values
    elif groups is not None:
        j = torch.as_tensor(same_group_partner(groups, rng), device=a.device)
        d_neg = (a - p[j]).norm(dim=-1)
    else:
        shift = int(rng.integers(1, n)) if n > 1 else 0
        d_neg = (a - p.roll(shift, dims=0)).norm(dim=-1)
    loss = F.relu(d_pos - d_neg + margin).mean()
    if z_shift is not None:
        zs = F.normalize(z_shift, dim=-1)                                    # [K, n, d]
        d_sh = (a[None] - zs).norm(dim=-1)                                   # [K, n]
        loss = loss + weight * F.relu(d_pos[None] - d_sh + margin).mean()
    return loss


def group_summaries(h, groups):
    """DGI summary per row: sigmoid(mean of the batch rows of the same group). groups None -> one summary."""
    if groups is None:
        return torch.sigmoid(h.mean(0)).expand_as(h)
    g = torch.as_tensor(np.asarray(groups), device=h.device)
    S = torch.empty_like(h)
    glob = torch.sigmoid(h.mean(0))
    for v in torch.unique(g):
        sel = g == v
        S[sel] = torch.sigmoid(h[sel].mean(0)) if sel.sum() > 0 else glob
    return S


def dgi_sr(disc_rows, h, h_neg, groups=None):
    """DGI binary cross-entropy. h [B, d] positives; h_neg [K*B, d] negatives (corrupted or shifted
    copies, K blocks in the order of h); disc_rows(h, S) -> logits per row. Positives and negatives
    weigh equally (as the original with K = 1)."""
    S = group_summaries(h, groups)
    K = h_neg.shape[0] // h.shape[0]
    pos = disc_rows(h, S)
    neg = disc_rows(h_neg, S.repeat(K, 1))
    if K == 1:   # exactly the original: one BCE over the concatenation
        logits = torch.cat([pos, neg])
        target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                  + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
