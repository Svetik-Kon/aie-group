from __future__ import annotations

from pathlib import Path
from pprint import pformat

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам;
    - эвристики качества (quality_flags).
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))

    # Покажем эвристики качества (чтобы было видно, что compute_quality_flags работает)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    typer.echo("\nQuality flags:")
    typer.echo(pformat(flags, sort_dicts=True))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, "--max-hist-columns", help="Максимум числовых колонок для гистограмм.", min=1
    ),
    top_k_categories: int = typer.Option(
        5, "--top-k-categories", help="Сколько top-значений выводить для категориальных колонок.", min=1
    ),
    title: str = typer.Option(
        "EDA-отчёт", "--title", help="Заголовок отчёта (первая строка report.md)."
    ),
    min_missing_share: float = typer.Option(
        0.2,
        "--min-missing-share",
        help="Порог доли пропусков, выше которого колонка считается проблемной.",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """
    Сгенерировать EDA-отчёт:
    - табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv;
    - report.md;
    - картинки: hist_*.png, missing_matrix.png, correlation_heatmap.png.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1) Базовые расчёты
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2) Качество данных
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3) Колонки с “проблемными” пропусками по порогу
    problem_missing_df = pd.DataFrame()
    if not missing_df.empty:
        problem_missing_df = (
            missing_df[missing_df["missing_share"] >= min_missing_share]
            .sort_values("missing_share", ascending=False)
        )

    # 4) Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)

    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)

    if not problem_missing_df.empty:
        problem_missing_df.to_csv(out_root / "problem_missing.csv", index=True)

    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)

    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5) Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Параметры отчёта\n\n")
        f.write(f"- Максимум гистограмм (числовых колонок): {max_hist_columns}\n")
        f.write(f"- Top-k категорий для категориальных признаков: {top_k_categories}\n")
        f.write(f"- Порог доли пропусков для проблемных колонок: {min_missing_share:.0%}\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Есть константные колонки: **{quality_flags['has_constant_columns']}**\n")
        f.write(f"- Есть high-cardinality категориальные: **{quality_flags['has_high_cardinality_categoricals']}**\n")
        f.write(f"- Есть дубликаты id-полей: **{quality_flags['has_suspicious_id_duplicates']}**\n\n")

        # Чтобы было видно, какие именно колонки попали под флаги
        if quality_flags.get("constant_columns"):
            f.write(f"- Константные колонки: `{', '.join(quality_flags['constant_columns'])}`\n")
        if quality_flags.get("high_cardinality_categoricals"):
            f.write(f"- High-cardinality: `{', '.join(quality_flags['high_cardinality_categoricals'])}`\n")
        if quality_flags.get("id_columns_with_duplicates"):
            f.write(f"- ID с дубликатами: `{', '.join(quality_flags['id_columns_with_duplicates'])}`\n")
        f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файл `missing.csv` и `missing_matrix.png`.\n\n")
            f.write(f"### Колонки с долей пропусков ≥ {min_missing_share:.0%}\n\n")
            if problem_missing_df.empty:
                f.write("Таких колонок нет.\n\n")
            else:
                for col_name, row in problem_missing_df.iterrows():
                    f.write(
                        f"- **{col_name}**: {row['missing_share']:.1%} "
                        f"({int(row['missing_count'])} шт.)\n"
                    )
                f.write("\nСм. также `problem_missing.csv`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"См. файлы `hist_*.png` (не более {max_hist_columns} числовых колонок).\n")

    # 6) Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    if (out_root / "problem_missing.csv").exists():
        typer.echo("- Дополнительно: problem_missing.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
