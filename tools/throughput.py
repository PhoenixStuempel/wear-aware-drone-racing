"""Sim speed on the GPU, in drone steps per second (1 step = 1 control step = 1/50 s of flight).

python -m tools.throughput
"""

import time

import torch

from sim.drones import Drones

STEPS = 200

for count in (1024, 4096, 16384, 65536):
    d = Drones(count, device="cuda")
    command = 0.35 + 0.2 * torch.rand(count, 4, device="cuda")
    for _ in range(20):  # warm up
        d.step(command)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(STEPS):
        d.step(command)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(
        f"{count:>6} drones: {count * STEPS / seconds / 1e6:6.2f} M steps/s  ({seconds / STEPS * 1000:.2f} ms per step)"
    )
