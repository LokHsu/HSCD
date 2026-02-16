import torch

def flatten_grads(grads, model):
    flat = []
    for g, p in zip(grads, model.parameters()):
        if g is None:
            flat.append(torch.zeros_like(p).view(-1))
        else:
            flat.append(g.contiguous().view(-1))
    return torch.cat(flat)

def get_task_gradients(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    return flatten_grads(grads, model)

def project_simplex(v, z=1.0):
    if v.sum() == z and torch.all(v >= 0):
        return v
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(len(v), device=v.device) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.clamp(v - theta, min=0)
    return w

def pareto_main_task_weights(g_rec, g_aux_list, lam=1.0):
    T = len(g_aux_list)

    # scale auxiliary gradients
    rec_norm = g_rec.norm() + 1e-12
    g_scaled = []
    for g in g_aux_list:
        g_norm = g.norm() + 1e-12
        g_scaled.append(g * (rec_norm / g_norm))
    G = torch.stack(g_scaled)

    A = G @ G.t()
    b = G @ g_rec

    w = torch.linalg.solve(A + 1e-8*torch.eye(T, device=A.device), -b)
    w = project_simplex(w, z=lam)
    return w.detach(), g_scaled

def pareto_backward(model, L_rec, aux_losses, lam=1.0):
    g_rec = get_task_gradients(L_rec, model)

    g_aux = []
    for L in aux_losses.values():
        g_aux.append(get_task_gradients(L, model))

    w, g_scaled = pareto_main_task_weights(g_rec, g_aux, lam)

    g_total = g_rec.clone()
    for wt, gt in zip(w, g_scaled):
        g_total += wt * gt

    idx = 0
    for p in model.parameters():
        if p.grad is None:
            numel = p.numel()
            p.grad = g_total[idx:idx+numel].view_as(p).clone()
            idx += numel
        else:
            numel = p.numel()
            p.grad.copy_(g_total[idx:idx+numel].view_as(p))
            idx += numel
    return w
