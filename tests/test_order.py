"""Rodas3's order of accuracy, measured on a problem with a known answer."""

import helpers
import numpy as np
import pytest
import torch

from maxichem import kincuda

pytestmark = pytest.mark.gpu

T_TEST = 1500.0  # K
P_TEST = 1e6  # dyn cm^-2
J_TEST = 1.0  # s^-1
END = 1.0  # s
STEPS = [4, 8, 16, 32]
ANY_ERROR_ACCEPTED = 1e6  # rtol: every try is accepted, so each call is one step of size dt
ORDER_BAND = (2.7, 3.3)  # around Rodas3's order 3: the slope varies this much at these step sizes


def test_rodas3_is_third_order(read_network, gpu):
    net = read_network("synthetic_decay")
    a = net.species.index("N2_1")
    errors = []
    with kincuda.CudaSolver(net, gpu, photo=True) as solver:
        for steps in STEPS:
            dt = END / steps
            one = torch.full((1,), 1.0, dtype=torch.float64, device=gpu)
            x = helpers.state(net, 1, gpu, N2_1=1.0)
            J = torch.full((1, len(net.photo)), J_TEST, dtype=torch.float64, device=gpu)
            for _ in range(steps):
                x, _, done, accepted = solver(
                    x, T_TEST * one, P_TEST * one, dt * one, dt * one, rtol=ANY_ERROR_ACCEPTED, J=J
                )
                assert bool(done.all())
                assert float(accepted[0]) == 1.0
            errors.append(abs(float(x[0, a]) - helpers.decay_answer(J_TEST * END)[0]))
    slope = np.polyfit(np.log(END / np.array(STEPS)), np.log(errors), 1)[0]
    print(f"order: errors {errors}, slope {slope:.3f}")
    assert ORDER_BAND[0] <= slope <= ORDER_BAND[1]
