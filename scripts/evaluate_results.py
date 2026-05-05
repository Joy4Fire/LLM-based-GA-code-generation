import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "ga_visagent"


def iter_result_files(results_dir: Path):
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"summary.json", "analysis_report.json", "regression_report.json"}:
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment JSON result files.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = list(iter_result_files(results_dir))
    total = len(files)
    success = 0
    failed = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("success") is True:
            success += 1
        else:
            failed += 1

    rate = (success / total * 100) if total else 0.0
    summary = {
        "results_dir": str(results_dir),
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": round(rate, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
