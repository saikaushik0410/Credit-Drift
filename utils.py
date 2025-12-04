import numpy as np

def calculate_psi(base, new, bins=10):
    base = np.array(base)
    new = np.array(new)

    breakpoints = np.linspace(min(base), max(base), bins + 1)

    base_counts = np.histogram(base, breakpoints)[0] + 1e-6
    new_counts = np.histogram(new, breakpoints)[0] + 1e-6

    base_perc = base_counts / base_counts.sum()
    new_perc = new_counts / new_counts.sum()

    psi = np.sum((new_perc - base_perc) * np.log(new_perc / base_perc))
    return float(psi)
