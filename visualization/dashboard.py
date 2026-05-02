from __future__ import annotations

import argparse
import base64
import io
import json
import textwrap
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def discover_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not ARTIFACTS_DIR.exists():
        return runs

    for summary_path in ARTIFACTS_DIR.rglob("summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        run_dir = summary_path.parent
        run_id = str(run_dir.relative_to(ARTIFACTS_DIR)).replace("\\", "/")
        predictions_path = run_dir / "predictions.csv"
        root_causes_path = run_dir / "root_causes.csv"
        segment_path = run_dir / "root_cause_segments.csv"
        event_match_path = run_dir / "root_cause_event_matches.csv"
        runs.append(
            {
                "id": run_id,
                "name": run_id,
                "summary": summary,
                "run_dir": run_dir,
                "predictions_path": predictions_path if predictions_path.exists() else None,
                "root_causes_path": root_causes_path if root_causes_path.exists() else None,
                "segment_path": segment_path if segment_path.exists() else None,
                "event_match_path": event_match_path if event_match_path.exists() else None,
            }
        )

    runs.sort(key=lambda item: item["id"])
    return runs


def discover_leaderboards() -> list[dict[str, Any]]:
    leaderboards: list[dict[str, Any]] = []
    if not ARTIFACTS_DIR.exists():
        return leaderboards

    for leaderboard_path in ARTIFACTS_DIR.rglob("leaderboard.csv"):
        try:
            table = pd.read_csv(leaderboard_path)
        except Exception:
            continue

        leaderboard_id = str(leaderboard_path.relative_to(ARTIFACTS_DIR)).replace("\\", "/")
        leaderboards.append(
            {
                "id": leaderboard_id,
                "path": leaderboard_path,
                "table": table,
                "name": leaderboard_id,
            }
        )

    leaderboards.sort(key=lambda item: item["id"])
    return leaderboards


def load_leaderboard(leaderboard_id: str | None) -> dict[str, Any] | None:
    leaderboards = discover_leaderboards()
    if not leaderboards:
        return None
    if leaderboard_id is None:
        return leaderboards[0]
    for item in leaderboards:
        if item["id"] == leaderboard_id:
            return item
    return leaderboards[0]


def load_run(run_id: str | None) -> dict[str, Any] | None:
    runs = discover_runs()
    if not runs:
        return None
    if run_id is None:
        return runs[-1]
    for run in runs:
        if run["id"] == run_id:
            return run
    return runs[-1]


def fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_score_chart(predictions: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    x = range(len(predictions))
    ax.plot(x, predictions["anomaly_score"], color="#1f77b4", linewidth=1.0, label="Anomaly score")

    if "label" in predictions.columns:
        anomaly_points = predictions[predictions["label"] == 1]
        ax.scatter(anomaly_points.index, anomaly_points["anomaly_score"], s=8, color="#d62728", label="Ground truth anomaly")

    predicted_points = predictions[predictions["prediction"] == 1]
    ax.scatter(predicted_points.index, predicted_points["anomaly_score"], s=8, color="#ff7f0e", alpha=0.6, label="Predicted anomaly")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    return fig_to_base64(fig)


def render_prediction_trend(predictions: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 2.2))
    x = range(len(predictions))
    ax.plot(x, predictions["prediction"], color="#ff7f0e", linewidth=1.0, label="Prediction")
    if "label" in predictions.columns:
        ax.plot(x, predictions["label"], color="#2ca02c", linewidth=1.0, alpha=0.75, label="Ground truth")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("0 / 1")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    return fig_to_base64(fig)


def render_bar_chart(table: pd.DataFrame, x_col: str, y_col: str, title: str, color: str = "#1f77b4") -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(table[x_col], table[y_col], color=color)
    ax.set_title(title)
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    return fig_to_base64(fig)


def render_comparison_chart(table: pd.DataFrame, title: str) -> str:
    numeric_columns = [column for column in ["precision", "recall", "f1", "roc_auc", "pr_auc", "rca_hit_at_5"] if column in table.columns]
    comparison = table.copy()
    comparison = comparison[comparison["status"] == "success"].copy()
    if comparison.empty or not numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No comparable successful runs", ha="center", va="center")
        ax.axis("off")
        return fig_to_base64(fig)

    label_column = "experiment_name" if "experiment_name" in comparison.columns else "config_path"
    comparison[label_column] = comparison[label_column].fillna(comparison["config_path"])
    melted = comparison[[label_column, *numeric_columns]].melt(id_vars=label_column, var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for metric_name, group in melted.groupby("metric"):
        ax.plot(group[label_column], group["value"], marker="o", linewidth=1.8, label=metric_name)

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=3, fontsize=9)
    return fig_to_base64(fig)


def metric_cards(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    cards = []
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        if key in metrics:
            cards.append(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{key.upper()}</div>
                    <div class="metric-value">{metrics[key]:.4f}</div>
                </div>
                """
            )
    rca_block = summary.get("rca", {})
    if isinstance(rca_block, dict) and rca_block.get("metrics"):
        for key, value in rca_block["metrics"].items():
            cards.append(
                f"""
                <div class="metric-card metric-card-alt">
                    <div class="metric-label">{key.upper()}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                """
            )
    return "\n".join(cards)


def dataframe_to_html(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "<p class='empty-note'>Không có dữ liệu để hiển thị.</p>"
    preview = frame.head(max_rows).copy()
    return preview.to_html(index=False, classes="data-table", border=0)


def leaderboard_to_html(frame: pd.DataFrame, runs: list[dict[str, Any]], max_rows: int = 30) -> str:
    if frame.empty:
        return "<p class='empty-note'>Khong co leaderboard de hien thi.</p>"

    preview = frame.head(max_rows).copy()
    run_ids = {run["id"] for run in runs}

    if "run_id" in preview.columns or "experiment_name" in preview.columns:
        preview["open_run"] = ""
        for index, row in preview.iterrows():
            candidate_run = None
            run_dir = row.get("run_dir")
            if isinstance(run_dir, str) and run_dir:
                run_path = Path(run_dir)
                try:
                    candidate_run = run_path.relative_to(ARTIFACTS_DIR).as_posix()
                except ValueError:
                    if run_path.parts and "artifacts" in run_path.parts:
                        artifacts_index = run_path.parts.index("artifacts")
                        candidate_run = Path(*run_path.parts[artifacts_index + 1 :]).as_posix()

            if candidate_run is None:
                config_path = str(row.get("config_path", ""))
                if "smd\\machine_1_1\\isolation_forest_percentile_97.json" in config_path:
                    candidate_run = "smd/machine-1-1/isolation_forest_percentile_97"
                elif "smd\\machine_1_1\\isolation_forest.json" in config_path:
                    candidate_run = "smd/machine-1-1/isolation_forest"
                elif "sklearn_breast_cancer\\isolation_forest.json" in config_path:
                    candidate_run = "sklearn_breast_cancer/isolation_forest"
                elif "credit_card\\isolation_forest.json" in config_path:
                    candidate_run = "credit_card/isolation_forest"

            if candidate_run and candidate_run in run_ids:
                preview.at[index, "open_run"] = f"<a href='/?run={candidate_run}'>Open</a>"

        ordered_columns = ["open_run"] + [column for column in preview.columns if column != "open_run"]
        preview = preview[ordered_columns]

    return preview.to_html(index=False, classes="data-table", border=0, escape=False)


def build_dashboard_html(run: dict[str, Any]) -> str:
    summary = run["summary"]
    predictions = pd.read_csv(run["predictions_path"]) if run["predictions_path"] else pd.DataFrame()
    root_causes = pd.read_csv(run["root_causes_path"]) if run["root_causes_path"] else pd.DataFrame()
    segment_rankings = pd.read_csv(run["segment_path"]) if run["segment_path"] else pd.DataFrame()
    event_matches = pd.read_csv(run["event_match_path"]) if run["event_match_path"] else pd.DataFrame()
    leaderboard = load_leaderboard(None)
    leaderboard_table = leaderboard["table"] if leaderboard is not None else pd.DataFrame()

    if root_causes.empty:
        rca_block = summary.get("rca")
        if isinstance(rca_block, list) and rca_block:
            root_causes = pd.DataFrame(rca_block)
        elif isinstance(rca_block, dict) and rca_block.get("global_ranking"):
            root_causes = pd.DataFrame(rca_block["global_ranking"])

    score_chart = render_score_chart(predictions, f"Anomaly Score Trend - {run['name']}") if not predictions.empty else ""
    prediction_chart = render_prediction_trend(predictions, f"Prediction vs Ground Truth - {run['name']}") if not predictions.empty else ""
    root_cause_chart = render_bar_chart(root_causes, "feature", "contribution_score", "Top Root Causes", "#d62728") if not root_causes.empty else ""
    comparison_chart = render_comparison_chart(leaderboard_table, "Leaderboard Comparison") if not leaderboard_table.empty else ""

    available_runs = "".join(
        f"<option value='/?run={candidate['id']}' {'selected' if candidate['id'] == run['id'] else ''}>{candidate['id']}</option>"
        for candidate in discover_runs()
    )

    available_leaderboards = "".join(
        f"<li><strong>{candidate['id']}</strong> - {len(candidate['table'])} rows</li>"
        for candidate in discover_leaderboards()
    )

    metadata_items = []
    for key, value in summary.get("dataset", {}).items():
        metadata_items.append(f"<li><strong>{key}</strong>: {value}</li>")

    detector_pretty = json.dumps(summary.get("detector", {}), ensure_ascii=False, indent=2)

    return f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="utf-8" />
        <title>Opstimus Dashboard</title>
        <style>
            :root {{
                --bg: #f4f1ea;
                --panel: #fffaf2;
                --ink: #1f1f1f;
                --muted: #6f6a61;
                --line: #d9cfbf;
                --accent: #8f3b2f;
                --accent-soft: #ead8d1;
                --accent-2: #275d63;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: Georgia, "Times New Roman", serif;
                background:
                    radial-gradient(circle at top left, #f9efe3 0, transparent 28%),
                    linear-gradient(180deg, #f4f1ea 0%, #efe7db 100%);
                color: var(--ink);
            }}
            .shell {{
                max-width: 1380px;
                margin: 0 auto;
                padding: 28px 24px 48px;
            }}
            .hero {{
                display: grid;
                grid-template-columns: 1.5fr 1fr;
                gap: 18px;
                align-items: stretch;
                margin-bottom: 22px;
            }}
            .hero-card, .panel {{
                background: rgba(255, 250, 242, 0.92);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 12px 30px rgba(74, 58, 34, 0.08);
            }}
            h1 {{
                margin: 0 0 8px;
                font-size: 34px;
                line-height: 1.1;
                letter-spacing: 0.02em;
            }}
            h2 {{
                margin: 0 0 14px;
                font-size: 22px;
                color: var(--accent);
            }}
            h3 {{
                margin: 0 0 10px;
                font-size: 18px;
                color: var(--accent-2);
            }}
            p, li {{
                font-size: 15px;
                line-height: 1.65;
            }}
            .muted {{
                color: var(--muted);
            }}
            .run-picker {{
                display: flex;
                gap: 10px;
                align-items: center;
                margin-top: 18px;
            }}
            select {{
                flex: 1;
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid var(--line);
                background: white;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 12px;
            }}
            .metric-card {{
                padding: 14px;
                border-radius: 14px;
                background: white;
                border: 1px solid #eadfd2;
            }}
            .metric-card-alt {{
                background: var(--accent-soft);
            }}
            .metric-label {{
                color: var(--muted);
                font-size: 12px;
                letter-spacing: 0.08em;
            }}
            .metric-value {{
                font-size: 28px;
                margin-top: 6px;
                color: var(--accent);
                font-weight: bold;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 18px;
                margin-top: 18px;
            }}
            .stack {{
                display: grid;
                gap: 18px;
            }}
            img {{
                width: 100%;
                border-radius: 12px;
                border: 1px solid #e3d8ca;
                background: white;
            }}
            ul {{
                padding-left: 18px;
                margin: 0;
            }}
            pre {{
                background: #27231f;
                color: #f6eee1;
                padding: 14px;
                border-radius: 12px;
                overflow: auto;
                font-size: 13px;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            .data-table th, .data-table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #e6dccd;
                text-align: left;
                vertical-align: top;
            }}
            .data-table th {{
                background: #f1e6d9;
                color: #5a392f;
            }}
            .empty-note {{
                color: var(--muted);
                font-style: italic;
            }}
            .footer-note {{
                margin-top: 18px;
                padding: 14px 16px;
                border-left: 4px solid var(--accent);
                background: #fff3ea;
                border-radius: 8px;
            }}
            @media (max-width: 980px) {{
                .hero, .grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <script>
            function handleRunChange(select) {{
                window.location.href = select.value;
            }}
        </script>
    </head>
    <body>
        <div class="shell">
            <section class="hero">
                <div class="hero-card">
                    <div class="muted">Opstimus Demo Dashboard</div>
                    <h1>Giám sát chỉ số, xu hướng bất thường và phân tích RCA trên cùng một web view</h1>
                    <p>
                        Dashboard này đọc trực tiếp các artifact đã sinh từ pipeline anomaly detection và RCA.
                        Bạn có thể dùng nó để demo luận văn: xem metric detection, trend anomaly score,
                        đối chiếu prediction với ground truth, và xem root cause ở mức toàn cục hoặc theo event.
                    </p>
                    <div class="run-picker">
                        <label for="run">Chọn lần chạy:</label>
                        <select id="run" onchange="handleRunChange(this)">
                            {available_runs}
                        </select>
                    </div>
                </div>
                <div class="hero-card">
                    <h2>Thông tin lần chạy</h2>
                    <ul>
                        {''.join(metadata_items)}
                    </ul>
                    <div class="footer-note">
                        Run hiện tại: <strong>{run['id']}</strong><br/>
                        Detector: <strong>{summary.get('detector', {}).get('name', 'unknown')}</strong>
                    </div>
                </div>
            </section>

            <section class="panel">
                <h2>Tổng quan chỉ số</h2>
                <div class="metrics">
                    {metric_cards(summary)}
                </div>
            </section>

            <section class="grid">
                <div class="panel">
                    <h2>Batch Leaderboards</h2>
                    <p class="muted">
                        Dashboard tự động quét mọi file <code>leaderboard.csv</code> trong <code>artifacts/</code> để phục vụ so sánh nhiều lần chạy.
                    </p>
                    <ul>
                        {available_leaderboards or "<li>Chưa có leaderboard nào.</li>"}
                    </ul>
                    <div class="footer-note">
                        Leaderboard mặc định đang hiển thị: <strong>{leaderboard['id'] if leaderboard else 'none'}</strong>
                    </div>
                </div>
                <div class="panel">
                    <h2>So sánh nhiều run</h2>
                    <img src="data:image/png;base64,{comparison_chart}" alt="Leaderboard comparison chart" />
                    {leaderboard_to_html(leaderboard_table, discover_runs(), max_rows=20)}
                </div>
            </section>

            <section class="grid">
                <div class="stack">
                    <div class="panel">
                        <h2>Xu hướng Anomaly Score</h2>
                        <img src="data:image/png;base64,{score_chart}" alt="Anomaly score chart" />
                    </div>
                    <div class="panel">
                        <h2>So sánh Prediction và Ground Truth</h2>
                        <img src="data:image/png;base64,{prediction_chart}" alt="Prediction trend chart" />
                    </div>
                    <div class="panel">
                        <h2>Segment-level RCA</h2>
                        <p class="muted">
                            Bảng dưới đây cho thấy top-k channel/feature đóng góp mạnh nhất cho từng đoạn bất thường liên tiếp.
                        </p>
                        {dataframe_to_html(segment_rankings, max_rows=30)}
                    </div>
                </div>

                <div class="stack">
                    <div class="panel">
                        <h2>Root Cause Toàn Cục</h2>
                        <img src="data:image/png;base64,{root_cause_chart}" alt="Root cause chart" />
                        {dataframe_to_html(root_causes, max_rows=10)}
                    </div>
                    <div class="panel">
                        <h2>RCA Event Matching</h2>
                        <p class="muted">
                            Dành cho SMD hoặc dataset có interpretation label. Bảng này giúp bạn demo việc root cause tìm được có trùng với ground truth event hay không.
                        </p>
                        {dataframe_to_html(event_matches, max_rows=20)}
                    </div>
                    <div class="panel">
                        <h2>Cấu hình Detector</h2>
                        <pre>{detector_pretty}</pre>
                    </div>
                </div>
            </section>
        </div>
    </body>
    </html>
    """


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        run_id = query.get("run", [None])[0]

        if run_id is None:
            default_run = load_run(None)
            if default_run is not None:
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", f"/?run={default_run['id']}")
                self.end_headers()
                return

        run = load_run(run_id)
        if run is None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Khong tim thay artifact nao trong thu muc artifacts/</h1></body></html>"
            )
            return

        html = build_dashboard_html(run)
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local dashboard for anomaly detection and RCA artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              venv_opstimus\\Scripts\\python.exe visualization\\dashboard.py --port 8765
            """
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the local dashboard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Opstimus dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
