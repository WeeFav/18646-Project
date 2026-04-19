from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.errors import EmptyDataError


def load_all_results(pattern='results*.csv'):
    csv_paths = sorted(Path('.').glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    data = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            print(f"Skipping {path.name}: file is empty")
            continue

        if df.empty:
            print(f"Skipping {path.name}: no rows")
            continue

        if 'Resolution' not in df.columns:
            print(f"Skipping {path.name}: missing column ['Resolution']")
            continue

        transform_candidates = [
            'Transform_ms',
            'All_Transform_ms',
            'All_Transform_Cycles',
        ]
        raster_candidates = [
            'Raster_ms',
            'Raster_Loop_ms',
            'Raster_Loop_Cycles',
        ]
        total_candidates = ['Total_ms']

        transform_col = next((c for c in transform_candidates if c in df.columns), None)
        raster_col = next((c for c in raster_candidates if c in df.columns), None)
        total_col = next((c for c in total_candidates if c in df.columns), None)

        if transform_col is None or raster_col is None or total_col is None:
            print(
                f"Skipping {path.name}: missing required metric columns "
                f"(found: {list(df.columns)})"
            )
            continue

        # Normalize metric names for downstream plotting.
        df = df.rename(columns={
            transform_col: 'Transform_Value',
            raster_col: 'Raster_Value',
            total_col: 'Total_Value',
        })

        averages = (
            df.groupby('Resolution')[['Transform_Value', 'Raster_Value', 'Total_Value']]
            .mean()
            .reset_index()
            .sort_values('Resolution')
        )

        label = path.stem.replace('results_', '')
        if label == 'results':
            label = 'default'

        data.append((label, averages))

    if not data:
        raise ValueError('No valid CSV files with required columns were found.')

    return data


def build_x_labels(resolutions):
    tri_counts = {
        16: '33,462',
        32: '133,020',
        64: '530,384',
        128: '2,117,776',
    }
    return [f"{res}\n({tri_counts.get(res, 'N/A')} triangles)" for res in resolutions]


def plot_metric(all_data, metric_col, title, out_name, color_map='tab10'):
    # Use union of all available resolutions so every file can be compared.
    all_resolutions = sorted(
        set().union(*[set(df['Resolution'].tolist()) for _, df in all_data])
    )
    x_labels = build_x_labels(all_resolutions)
    x = np.arange(len(all_resolutions))

    n_series = len(all_data)
    width = 0.8 / max(1, n_series)

    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap(color_map)

    for idx, (label, df) in enumerate(all_data):
        value_map = dict(zip(df['Resolution'], df[metric_col]))
        values = [value_map.get(r, np.nan) for r in all_resolutions]
        offset = (idx - (n_series - 1) / 2) * width
        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=cmap(idx % 10),
            edgecolor='black',
            linewidth=0.7,
        )

        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{val:.2e}',
                va='bottom',
                ha='center',
                fontsize=8,
                rotation=90,
            )

    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Resolution Scale and Triangle Count', fontsize=12)
    plt.ylabel('Average Time (ms)', fontsize=12)
    plt.xticks(x, x_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Result File', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    print(f"Saved {title} plot to {out_name}")
    plt.close()


def main():
    try:
        all_data = load_all_results('results*.csv')
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    print('Loaded result files:')
    for label, df in all_data:
        print(f"  - {label}: {len(df)} resolution point(s)")

    plot_metric(
        all_data,
        metric_col='Transform_Value',
        title='Average Execution Time: Vertex Transform (All CSVs)',
        out_name='vertex_transform_all_results.png',
    )

    plot_metric(
        all_data,
        metric_col='Raster_Value',
        title='Average Execution Time: Rasterization Loop (All CSVs)',
        out_name='rasterization_all_results.png',
    )


if __name__ == '__main__':
    main()