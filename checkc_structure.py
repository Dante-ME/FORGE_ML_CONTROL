from pathlib import Path

ROOT = Path(__file__).resolve().parent

required_dirs = [
    "data",
    "data/raw",
    "data/processed",
    "data/synthetic",

    "outputs",
    "outputs/figures",
    "outputs/figures/stage1",
    "outputs/figures/stage4",
    "outputs/tables",
    "outputs/tables/stage1",
    "outputs/tables/stage4",
    "outputs/models",
    "outputs/models/stage1",

    "src",
    "src/stage1_diagnostic",
    "src/stage4_control",
    "src/gui",

    "notebooks",
    "reports",
]

required_files = [
    "requirements.txt",
    "README.md",
    "main.py",

    "src/stage1_diagnostic/__init__.py",
    "src/stage1_diagnostic/generate_dataset.py",
    "src/stage1_diagnostic/preprocess.py",
    "src/stage1_diagnostic/train_xgboost.py",
    "src/stage1_diagnostic/evaluate.py",
    "src/stage1_diagnostic/predict.py",

    "src/stage4_control/__init__.py",
    "src/stage4_control/generate_profile.py",
    "src/stage4_control/fuzzy_controller.py",
    "src/stage4_control/rule_based_controller.py",
    "src/stage4_control/simulate_baseline.py",
    "src/stage4_control/simulate_proposed.py",
    "src/stage4_control/evaluate_control.py",

    "src/gui/__init__.py",
    "src/gui/app.py",

    "reports/methodology_notes.md",
    "reports/stage1_results_summary.md",
    "reports/stage4_results_summary.md",
]


def check_paths(paths, path_type):
    missing = []
    existing = []

    for path in paths:
        full_path = ROOT / path
        if path_type == "dir":
            ok = full_path.is_dir()
        else:
            ok = full_path.is_file()

        if ok:
            existing.append(path)
        else:
            missing.append(path)

    return existing, missing


def main():
    print("=" * 60)
    print("FORGE Project Structure Checker")
    print("=" * 60)

    existing_dirs, missing_dirs = check_paths(required_dirs, "dir")
    existing_files, missing_files = check_paths(required_files, "file")

    print(f"\nExisting directories: {len(existing_dirs)}/{len(required_dirs)}")
    print(f"Existing files      : {len(existing_files)}/{len(required_files)}")

    if missing_dirs:
        print("\nMissing directories:")
        for path in missing_dirs:
            print(f"  [DIR]  {path}")
    else:
        print("\nNo missing directories.")

    if missing_files:
        print("\nMissing files:")
        for path in missing_files:
            print(f"  [FILE] {path}")
    else:
        print("\nNo missing files.")

    print("\n" + "=" * 60)

    if not missing_dirs and not missing_files:
        print("STATUS: Project structure is COMPLETE.")
    else:
        print("STATUS: Project structure is INCOMPLETE.")
        print("Please create the missing folders/files above.")

    print("=" * 60)


if __name__ == "__main__":
    main()