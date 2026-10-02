"""
losses.py
Contrastive objectives for CoLA-style graph anomaly detection, including the
proposed structure-graded loss (Notion: "Loss Function Design (draft)").

Notation
  s(v, G)  discriminator score: high = "node v fits subgraph G"
           (BilinearDiscriminator: h_v^T W r(G), as in CoLA)
  pos      s(v, G_v)       for B anchors                     shape [B]
  neg      s(v, G_{u_k})   for K negatives per anchor        shape [B, K]
  dist     d(v, u_k) in [0, 1], structural distance          shape [B, K]

Structural distance (label-blind, uses declared dependencies only)
  d(v, u) = 1 - |D(v) & D(u)| / |D(v) | D(u)|,   d = 1 if either set is empty

Structure-graded loss
  L_v = -log  exp(pos/tau) / ( exp(pos/tau) + sum_k exp((neg_k + m0 * d_k)/tau) )

Special cases (for the ablation)
  m0 = 0                         -> InfoNCE                (infonce_loss)
  K = 1, small tau               -> triplet with a graded margin (graded_triplet_loss)
  K = 1, m0 = 0, BCE per pair    -> CoLA's objective       (cola_bce_loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearDiscriminator(nn.Module):
    """CoLA's discriminator: s(v, G) = h_v^T W r(G)."""

    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, h, r):
        """h: [..., dim] node embeddings, r: [..., dim] subgraph summaries."""
        return self.bilinear(h, r).squeeze(-1)


def dependency_distance(dep_sets, anchors, negatives):
    """d(v, u) = 1 - Jaccard(D(v), D(u)); 1 when either set is empty.

    dep_sets:  list of sets, dependency targets per node (any hashable ids)
    anchors:   [B] node indices
    negatives: [B, K] node indices of the negatives' centre nodes
    returns    float tensor [B, K]
    """
    anchors = torch.as_tensor(anchors).tolist()
    negatives = torch.as_tensor(negatives)
    B, K = negatives.shape
    out = torch.ones(B, K)
    for b, v in enumerate(anchors):
        dv = dep_sets[v]
        if not dv:
            continue
        for k, u in enumerate(negatives[b].tolist()):
            du = dep_sets[u]
            if du:
                out[b, k] = 1.0 - len(dv & du) / len(dv | du)
    return out


def structure_graded_loss(pos, neg, dist, tau=0.5, m0=1.0):
    """Proposed loss: InfoNCE whose margin for each negative grows with its
    structural distance, m(d) = m0 * d. Mean over anchors."""
    if pos.dim() != 1 or neg.dim() != 2 or neg.shape != dist.shape or neg.size(0) != pos.size(0):
        raise ValueError(f"shapes: pos {tuple(pos.shape)}, neg {tuple(neg.shape)}, dist {tuple(dist.shape)}")
    logits = torch.cat([pos.unsqueeze(1), neg + m0 * dist.to(neg)], dim=1) / tau
    target = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    return F.cross_entropy(logits, target)


def infonce_loss(pos, neg, tau=0.5):
    """Plain InfoNCE: every negative equally wrong (m0 = 0)."""
    return structure_graded_loss(pos, neg, torch.zeros_like(neg), tau=tau, m0=0.0)


def graded_triplet_loss(pos, neg, dist, m0=1.0):
    """Hinge version: each negative must score at least m0 * d below the positive."""
    return F.relu(neg + m0 * dist.to(neg) - pos.unsqueeze(1)).mean()


def cola_bce_loss(pos, neg):
    """CoLA's objective: binary cross-entropy, positives -> 1, negatives -> 0."""
    neg = neg.reshape(-1)
    logits = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return F.binary_cross_entropy_with_logits(logits, labels)


def anomaly_score(pos_rounds, neg_rounds):
    """CoLA's test-time score over R rounds of subgraph sampling:
    mean_r ( s(v, G_u^(r)) - s(v, G_v^(r)) ), random negatives u.
    Both inputs [R, N]; returns [N]. Higher = more anomalous."""
    return (neg_rounds - pos_rounds).mean(dim=0)
