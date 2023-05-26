import torch

def rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty(shape, device=device)
    x.random_(0, 2)  # Creates random tensor of 0s and 1s
    x[x == 0] = -1  # Turn the 0s into -1s
    return x

def hessian_penalty(G, z, G_z, k=2, epsilon=1, reduction=torch.max):
    rademacher_size = torch.Size((k, *z[0].size()))  # (k, N, z.size())
    xs = epsilon * rademacher(rademacher_size, device=G.device)
    second_orders = []
    for x in xs:  # Iterate over each (N, z.size()) tensor in xs
        central_second_order = multi_layer_second_directional_derivative(G, z, x, G_z, epsilon)
        second_orders.append(central_second_order)  # Appends a tensor with shape equal to G(z).size()
    loss = multi_stack_var_and_reduce(second_orders, reduction)  # (k, G(z).size()) --> scalar
    return loss


def ortho_jacob(G, z, G_z, k=2, epsilon=1, reduction=torch.max):
    rademacher_size = torch.Size((k, *z[0].size()))  # (k, N, z.size())
    xs = epsilon * rademacher(rademacher_size, device=G.device)
    first_orders = []
    for x in xs:  # Iterate over each (N, z.size()) tensor in xs
        first_order = multi_layer_first_directional_derivative(G, z, x, G_z, epsilon)
        first_orders.append(first_order)  # Appends a tensor with shape equal to G(z).size()
    loss = multi_stack_var_and_reduce(first_orders, reduction)  # (k, G(z).size()) --> scalar
    return loss

def multi_layer_second_directional_derivative(G, z, x, G_z, epsilon):
    """Estimates the second directional derivative of G w.r.t. its input at z in the direction x"""
    G_to_x ,   _  = G( z_ele + x for z_ele in z )
    G_from_x , _  = G( z_ele - x for z_ele in z )

    G_to_x = listify(G_to_x)
    G_from_x = listify(G_from_x)
    G_z = listify(G_z)

    eps_sqr = epsilon ** 2
    sdd = [(G2x - 2 * G_z_base + Gfx) / eps_sqr for G2x, G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd

def multi_layer_first_directional_derivative(G, z, x, G_z, epsilon):
    """Estimates the first directional derivative of G w.r.t. its input at z in the direction x"""
    G_to_x ,   _  = G( z_ele + x for z_ele in z )
    G_to_x = listify(G_to_x)
    G_z = listify(G_z)

    fdd = [(G2x - G_z_base) / epsilon for G2x, G_z_base in zip(G_to_x, G_z)]
    return fdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    """Equation (5) from the paper."""
    second_orders = torch.stack(list_of_activations)  # (k, N, C, H, W)
    var_tensor = torch.var(second_orders, dim=0, unbiased=True)  # (N, C, H, W)
    penalty = reduction(var_tensor)  # (1,) (scalar)
    return penalty



def multi_stack_var_and_reduce(sdds, reduction=torch.max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (5) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]
