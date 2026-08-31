import torch
import torch.nn.functional as F


def g_loss(causal_logits, residual_logits, label, alpha: float, beta: float):
    discriminative = F.cross_entropy(causal_logits, label)
    uniform = torch.full_like(residual_logits, 1 / residual_logits.shape[-1])
    residual = F.kl_div(F.log_softmax(residual_logits, -1), uniform, reduction="batchmean")
    orthogonal = F.cosine_similarity(causal_logits.flatten(), residual_logits.flatten(), dim=0)
    return discriminative + alpha * residual + beta * orthogonal