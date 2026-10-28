"""Command line: capability studies, approval decisions, UC-Cap scores and monitoring for a measurement file.

    uccap measurements.xlsx                      # writes measurements_uccap.html next to the file
    uccap measurements.csv --excel --cost-ratio 19
    uccap data.xlsx --sheet Line2 --out results/ --no-hybrid
    uccap data.xlsx --plots all                  # PNG figures of every dimension, in data_uccap_plots/
    uccap plan 1.5 1.67 2.0 --cost-ratio 19      # parts needed for each assumed true Cpk
    uccap monitor production.xlsx --plots        # acceptability monitoring after approval
    uccap --cite                                 # how to cite the articles behind each module

The file is read with :func:`uccap.read_table` (specification rows above the measurements,
or any layout it recognises) and analysed with :func:`uccap.batch_study`.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Optional, Sequence

KINDS = ("overview", "outliers", "stability", "capability", "approval", "uccap")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uccap", description="Process capability analysis, risk-calibrated approval "
                                "and hybrid statistical learning decision support for every dimension of a measurement file.")
    p.add_argument("data", nargs="?", help="CSV, TXT or Excel file in the standard layout")
    p.add_argument("--sheet", default=0, help="Excel sheet name or index (default: the first)")
    p.add_argument("--out", help="output folder or file stem (default: next to the data file)")
    p.add_argument("--html", action="store_true", help="write the coloured HTML report (default when no format is given)")
    p.add_argument("--excel", action="store_true", help="write the coloured Excel workbook (needs openpyxl)")
    p.add_argument("--csv", action="store_true", help="write the results table as CSV")
    p.add_argument("--c0", type=float, default=1.33, help="required capability C0 (default 1.33)")
    risk = p.add_mutually_exclusive_group()
    risk.add_argument("--alpha", type=float, help="tolerated probability of a false approval (default 0.05)")
    risk.add_argument("--cost-ratio", type=float, help="cost of a false approval / cost of a false rejection")
    p.add_argument("--no-hybrid", action="store_true", help="skip the UC-Cap risk score")
    p.add_argument("--model", metavar="PATH", help="trained UC-Cap parameters (JSON); not included in this release, "
                                                    "available from the authors on request")
    p.add_argument("--zero-lsl-limit", action="store_true",
                   help="read nominal 0 with LSL 0 as a real lower limit, not as a GD&T bound (upper limit only)")
    p.add_argument("--plots", nargs="?", const="overview", metavar="KINDS",
                   help="write PNG figures of every dimension: overview (default), outliers, stability, "
                        "capability, approval, uccap, a comma-separated list, or all (needs matplotlib)")
    p.add_argument("--show-values", action="store_true", help="write the measured values on the outlier and stability charts")
    p.add_argument("--quiet", action="store_true", help="do not print the summary table")
    p.add_argument("--cite", action="store_true", help="print how to cite uccap and the articles, then exit")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    return p


def _plan_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uccap plan", description="Number of parts needed so that a process with an "
                                "assumed true Cpk is accepted by the risk-calibrated rule with a wanted probability.")
    p.add_argument("expected", nargs="+", type=float, help="assumed true Cpk values, for example 1.5 1.67 2.0")
    p.add_argument("--c0", type=float, default=1.33, help="required capability C0 (default 1.33)")
    risk = p.add_mutually_exclusive_group()
    risk.add_argument("--alpha", type=float, help="tolerated probability of a false approval (default 0.05)")
    risk.add_argument("--cost-ratio", type=float, help="cost of a false approval / cost of a false rejection")
    p.add_argument("--power", type=float, default=0.8, help="wanted probability of acceptance (default 0.8)")
    p.add_argument("--parts", type=int, help="a planned number of parts: adds P(accept) and the smallest accepted estimate at it")
    p.add_argument("--plot", metavar="FILE", help="write the acceptance curves as PNG (needs matplotlib)")
    return p


def _plan(argv: Sequence[str]) -> int:
    from uccap.risk.planning import _alpha, acceptance_probability, min_accepted, sample_size
    args = _plan_parser().parse_args(argv)
    alpha = _alpha(args.alpha, args.cost_ratio)
    rule = f"cost ratio {args.cost_ratio:g}, alpha = {alpha:.4f}" if args.cost_ratio is not None else f"alpha = {alpha:.4g}"
    print(f"C0 = {args.c0:g}, {rule}, wanted P(accept) = {args.power:.0%}\n")
    head = f"{'true Cpk':>9} {'parts needed':>13} {'P(accept)':>10} {'smallest accepted':>18}"
    if args.parts:
        head += f" {'P(accept) at ' + str(args.parts):>17}"
    print(head)
    for c in args.expected:
        plan = sample_size(c, c0=args.c0, alpha=alpha, power=args.power)
        if plan.n is None:
            line = f"{c:>9g} {'not reachable':>13} {'':>10} {'':>18}"
        else:
            line = f"{c:>9g} {plan.n:>13d} {plan.achieved:>10.4f} {plan.threshold:>18.4f}"
        if args.parts:
            line += f" {acceptance_probability(c, args.parts, c0=args.c0, alpha=alpha):>17.4f}"
        print(line)
    if args.parts:
        t = min_accepted(args.parts, c0=args.c0, alpha=alpha)
        shown = "none (too few parts)" if t == float("inf") else f"{t:.4f}"
        print(f"\nWith {args.parts} parts the smallest accepted estimate is {shown}.")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from uccap.plot import planning_plot
        top = max([args.parts or 0] + [sample_size(c, c0=args.c0, alpha=alpha, power=args.power).n or 0
                                         for c in args.expected])
        ax = planning_plot(args.expected, c0=args.c0, alpha=alpha, power=args.power,
                           n_range=(8, int(min(max(60, 1.3 * top), 1000))))
        ax.figure.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"written: {args.plot}")
    return 0


def _monitor_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uccap monitor", description="Operational acceptability monitoring after "
                                "approval: the posterior probability that each dimension still meets its "
                                "conformance target, subgroup by subgroup, and its zone (continue, monitor, intervene).")
    p.add_argument("data", help="CSV, TXT or Excel file in the standard layout, rows in production order")
    p.add_argument("--sheet", default=0, help="Excel sheet name or index (default: the first)")
    p.add_argument("--out", help="output folder or file stem (default: next to the data file)")
    p.add_argument("--html", action="store_true", help="write the coloured HTML table (default when no format is given)")
    p.add_argument("--excel", action="store_true", help="write the coloured Excel workbook with every trajectory (needs openpyxl)")
    p.add_argument("--csv", action="store_true", help="write the results table and the trajectories as CSV")
    p.add_argument("--subgroup-size", type=int, default=5, help="consecutive values per subgroup (default 5)")
    p.add_argument("--q0", type=float, default=0.997, help="conformance target q0 (default 0.997)")
    p.add_argument("--gamma-a", type=float, default=0.95, help="continue when pi >= gamma_A (default 0.95)")
    p.add_argument("--gamma-r", type=float, default=0.80, help="intervene when pi <= gamma_R (default 0.80)")
    p.add_argument("--forgetting", type=float, default=0.95, help="forgetting factor (default 0.95)")
    p.add_argument("--t-min", type=int, default=10, help="first subgroup at which intervention is allowed (default 10)")
    p.add_argument("--draws", type=int, default=2000, help="posterior draws per subgroup (default 2000)")
    p.add_argument("--zero-lsl-limit", action="store_true",
                   help="read nominal 0 with LSL 0 as a real lower limit, not as a GD&T bound (upper limit only)")
    p.add_argument("--plots", action="store_true", help="write the overview and one trajectory figure per dimension "
                   "as PNG (needs matplotlib)")
    p.add_argument("--quiet", action="store_true", help="do not print the summary table")
    return p


def _monitor(argv: Sequence[str]) -> int:
    import uccap
    args = _monitor_parser().parse_args(argv)
    src = pathlib.Path(args.data)
    if not src.exists():
        print(f"uccap: {src} not found", file=sys.stderr)
        return 2
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    table = uccap.read_table(src, sheet=sheet)
    result = uccap.batch_monitor(table, subgroup_size=args.subgroup_size, q0=args.q0, gamma_a=args.gamma_a,
                                 gamma_r=args.gamma_r, forgetting=args.forgetting, t_min=args.t_min,
                                 n_draws=args.draws, zero_lsl_gdt=not args.zero_lsl_limit)
    stem = _stem(src, args.out, "monitor")
    written = []
    if args.html or not (args.excel or args.csv):
        result.to_html(str(stem.with_suffix(".html")), title=f"Acceptability monitoring: {src.name}")
        written.append(stem.with_suffix(".html"))
    if args.excel:
        result.to_excel(str(stem.with_suffix(".xlsx")))
        written.append(stem.with_suffix(".xlsx"))
    if args.csv:
        result.frame.to_csv(stem.with_suffix(".csv"), index=False)
        traj = stem.with_name(stem.name + "_trajectories.csv")
        result.trajectories(decimals=None).to_csv(traj, index=False)
        written += [stem.with_suffix(".csv"), traj]
    figures = []
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        folder = stem.with_name(stem.name + "_plots")
        folder.mkdir(parents=True, exist_ok=True)
        figs = [("all_dimensions.png", result.plot)] if result.results else []
        figs += [(f"{_safe(d)}_monitor.png", r.plot) for d, r in result.results.items()]
        for name, draw in figs:
            fig = draw()
            fig.savefig(folder / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append(folder / name)
    if not args.quiet:
        cols = [c for c in ("dimension", "subgroups", "Ppk_normal", "pi_final", "pi_min", "first_intervention",
                            "zone", "category") if c in result.frame.columns]
        print(result.table(cols).to_string(index=False))
        print("\n" + "   ".join(f"{k}: {v}" for k, v in result.frame["zone"].value_counts().to_dict().items()))
    for w in written:
        print(f"written: {w}")
    if figures:
        print(f"written: {len(figures)} figures in {folder}")
    return 0


def _stem(src: pathlib.Path, out: Optional[str], suffix: str) -> pathlib.Path:
    if out:
        o = pathlib.Path(out)
        stem = o / f"{src.stem}_{suffix}" if o.is_dir() or out.endswith(("/", "\\")) else o
    else:
        stem = src.with_name(f"{src.stem}_{suffix}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "dimension"


def _kinds(spec: str) -> list:
    kinds = list(KINDS) if spec == "all" else [k.strip() for k in spec.split(",") if k.strip()]
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise SystemExit(f"uccap: unknown plot kind {', '.join(bad)}; choose from {', '.join(KINDS)} or all")
    return kinds


def _write_plots(result, kinds, folder: pathlib.Path, show_values: bool) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from uccap import plot as P
    folder.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    def save(fig, name):
        path = folder / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if result.studies:
        save(result.plot().figure, "all_dimensions.png")
    for dim, study in result.studies.items():
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(dim)).strip("_") or "dimension"
        decision = result.decisions.get(dim)
        for kind in kinds:
            if kind == "uccap":
                risk = result.risks.get(dim)
                if risk is None:
                    skipped.append(f"{dim} uccap")
                    continue
                fig = P.uccap_figure(risk, title=str(dim))
            elif kind == "approval" and decision is None:
                skipped.append(f"{dim} approval")
                continue
            else:
                fig = study.plot(decision, title=str(dim), kind=kind, show_data=show_values)
            save(fig, f"{safe}_{kind}.png")
    for d, err in result.errors.items():
        skipped.append(f"{d} ({err})")
    if skipped:
        print("no figure: " + "; ".join(skipped))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    import uccap
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["plan"]:
        return _plan(argv[1:])
    if argv[:1] == ["monitor"]:
        return _monitor(argv[1:])
    args = _parser().parse_args(argv)
    if args.version:
        print(uccap.__version__)
        return 0
    if args.cite:
        print(uccap.cite())
        return 0
    if not args.data:
        _parser().print_usage()
        return 2
    src = pathlib.Path(args.data)
    if not src.exists():
        print(f"uccap: {src} not found", file=sys.stderr)
        return 2
    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    table = uccap.read_table(src, sheet=sheet)
    if args.model:
        uccap.hybrid.set_model_path(args.model)
    result = uccap.batch_study(table, c0=args.c0, alpha=args.alpha, cost_ratio=args.cost_ratio,
                               hybrid=not args.no_hybrid, zero_lsl_gdt=not args.zero_lsl_limit)
    if args.out:
        out = pathlib.Path(args.out)
        stem = out / f"{src.stem}_uccap" if out.is_dir() or args.out.endswith(("/", "\\")) else out
    else:
        stem = src.with_name(f"{src.stem}_uccap")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if args.html or not (args.excel or args.csv):
        result.to_html(str(stem.with_suffix(".html")), title=f"Capability results: {src.name}")
        written.append(stem.with_suffix(".html"))
    if args.excel:
        result.to_excel(str(stem.with_suffix(".xlsx")))
        written.append(stem.with_suffix(".xlsx"))
    if args.csv:
        result.frame.to_csv(stem.with_suffix(".csv"), index=False)
        written.append(stem.with_suffix(".csv"))
    figures = []
    if args.plots:
        folder = stem.with_name(stem.name + "_plots")
        figures = _write_plots(result, _kinds(args.plots), folder, args.show_values)
    if not args.quiet:
        cols = [c for c in ("dimension", "n", "normality", "Ppk", "p_fail", "threshold_decision", "decision",
                            "risk_score", "risk_level") if c in result.frame.columns]
        print(result.frame[cols].to_string(index=False))
        counts = result.frame["decision"].value_counts().to_dict() if "decision" in result.frame else {}
        print("\n" + "   ".join(f"{k}: {v}" for k, v in counts.items()))
    for w in written:
        print(f"written: {w}")
    if figures:
        print(f"written: {len(figures)} figures in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
