from pathlib import Path
import subprocess


def run_check_and_fix(target_id: int, base_dir: Path, spec_path: Path, report_path: Path | None) -> None:
    """Run ``check_and_fix_csv.py`` for the specified target.

    The original implementation hard-coded Windows style paths and mutated a
    global command list in-place, which made the script difficult to maintain
    and even introduced an unterminated string literal in the repository.  This
    helper centralises the command construction, keeps the paths
    platform-independent, and allows us to fail fast when the subprocess exits
    with an error.
    """

    input_csv = base_dir / f"C{target_id:02d}.csv"
    output_csv = base_dir / f"C{target_id:02d}_fix.csv"

    command = [
        "python",
        "util/check_and_fix_csv.py",
        str(input_csv),
        str(spec_path),
        str(output_csv),
    ]

    if report_path is not None:
        command.extend(["--report", str(report_path)])

    subprocess.run(command, check=True)
    print(f"check_and_fix for C{target_id:02d} completed")


def main() -> None:
    base_dir = Path("out") / "PWSCUP2025_Pre_Data_for_Attack"
    spec_path = Path("data") / "pre_columns_range.json"
    report_path = Path("fix_report.csv")

    for team in range(1, 21):
        run_check_and_fix(team, base_dir, spec_path, report_path)

    # C21 is intentionally skipped in the original script; keep the same
    # behaviour and handle C22 separately.
    run_check_and_fix(22, base_dir, spec_path, report_path)


if __name__ == "__main__":
    main()
