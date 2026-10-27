"""How many parts does an approval study need? (risk-calibrated rule, C0 = 1.33, lambda = 19)"""
from uccap import sample_size
from uccap.risk import acceptance_probability, min_accepted

print(sample_size(1.67, c0=1.33, cost_ratio=19, power=0.8).report())

print("\nparts needed for 80 % probability of approval")
for cpk in (1.5, 1.67, 2.0):
    print(f"  true Cpk {cpk:<5} {sample_size(cpk, cost_ratio=19).n:>4}")

print("\nwith 30 parts")
print(f"  smallest accepted estimate     {min_accepted(30, cost_ratio=19):.3f}")
print(f"  P(approval) if true Cpk = 1.67 {acceptance_probability(1.67, 30, cost_ratio=19):.1%}")

try:
    import matplotlib.pyplot as plt
    from uccap.plot import planning_plot
    planning_plot(cost_ratio=19, n_range=(8, 300))
    plt.savefig("planning.png", dpi=150)
    print("\nplanning.png written")
except ImportError:
    pass
