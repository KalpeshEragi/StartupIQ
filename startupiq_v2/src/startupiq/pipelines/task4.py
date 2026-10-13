"""Task 4 runner: raw -> cleaned CSVs + logs + figures.   PYTHONPATH=src python -m startupiq.pipelines.task4"""
import os

from startupiq.clean import formd, yc
from startupiq.eda import plots

RAW_YC = "data/raw/yc/all.json"
RAW_FD = "data/raw/formd"
PROC = "data/processed"
RES = "docs/results/task4"
FIG = os.path.join(RES, "figures")


def main():
    os.makedirs(PROC, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    ydf, ylog = yc.clean_yc(yc.load_raw(RAW_YC))
    yc.to_csv(ydf, os.path.join(PROC, "yc_clean.csv"))
    ylog.write(os.path.join(RES, "cleaning_log_yc.json"))
    fraw = formd.load_raw(RAW_FD)
    fdf, flog = formd.clean_formd(fraw)
    fdf.to_csv(os.path.join(PROC, "formd_clean.csv"), index=False)
    flog.write(os.path.join(RES, "cleaning_log_formd.json"))
    plots.yc_overview(ydf, FIG)
    plots.yc_text_by_industry(ydf, FIG)
    plots.tag_cooccurrence(ydf, FIG)
    plots.yc_team_size(ydf, FIG)
    plots.missingness(ydf, FIG, "yc_missingness.png", "YC: missing values after cleaning",
                      ["industry", "team_size", "launched_at", "batch_year", "domain", "usable_text", "n_tags"])
    plots.cleaning_table(ylog, FIG, "yc_cleaning_table.png", "YC cleaning log")
    plots.formd_overview(fdf, FIG)
    plots.formd_revenue_age(fdf, FIG)
    plots.missingness(fdf, FIG, "formd_missingness.png", "Form D: missing values after cleaning")
    plots.cleaning_table(flog, FIG, "formd_cleaning_table.png", "Form D cleaning log")
    print("YC", ydf.shape, "FormD", fdf.shape)


if __name__ == "__main__":
    main()
