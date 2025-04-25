#!/usr/bin/env python3

from pathlib import Path
from datetime import timedelta
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)

BLUESKY_CSV = DATA_DIR / "dataset.csv"
POLLS_CSV = DATA_DIR / "polls.csv"

LOG_PATH = OUT_DIR / "runtime.log"

logging.basicConfig(
    filename=LOG_PATH, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

WINDOWS = [
    ("2024-08-16", "2024-08-21"),
    ("2024-08-22", "2024-09-04"),
    ("2024-09-05", "2024-09-11"),
    ("2024-09-12", "2024-09-18"),
    ("2024-09-19", "2024-09-25"),
    ("2024-09-26", "2024-10-02"),
    ("2024-10-03", "2024-10-04"),
    ("2024-10-05", "2024-10-06"),
]
CANDIDATES = [
    "Ricardo Nunes", "Guilherme Boulos", "Pablo Marçal",
    "Tabata Amaral", "José Luiz Datena",
]
POLL_COL = {
    "Ricardo Nunes":   "ricardo_nunes",
    "Guilherme Boulos": "guilherme_boulos",
    "Pablo Marçal":     "pablo_marcal",
    "Tabata Amaral":    "tabata_amaral",
    "José Luiz Datena": "jose_luiz_datena"
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os dados da predição e das pesquisas eleitorais"""
    df = pd.read_csv(BLUESKY_CSV, parse_dates=["date"])
    polls = pd.read_csv(POLLS_CSV, parse_dates=["date"])
    return df, polls


def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Atribui o número da janela (1-8) a cada publicação.
    """
    df["date"] = (
        pd.to_datetime(df["date"], format="mixed")
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )

    bins = []
    for start, end in WINDOWS:
        bins.append(pd.Timestamp(start))
    bins.append(pd.Timestamp(WINDOWS[-1][1]) + timedelta(days=1))

    # 3️⃣  aplica corte
    df["window"] = pd.cut(
        df["date"],
        bins=bins,
        right=False,
        labels=range(1, len(WINDOWS) + 1)
    ).astype("Int64")

    return df


def plot_sentiment_series(df, path):
    """
    Plota a série temporal diária do índice de sentimento com faixas de janelas.
    """
    # Agrega diariamente
    daily = (
        df.assign(pos=lambda d: (d["predicted_class"] == 1).astype(int))
        .groupby([pd.Grouper(key="date", freq="D"), "subject"])
        .agg(sent_index=("pos", "mean"))
        .reset_index()
    )
    export_csv(daily, "sentiment_daily")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=daily, x="date", y="sent_index",
                 hue="subject", marker="o", ax=ax)

    # Faixas das janelas
    for i, (start, end) in enumerate(WINDOWS):
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
        color = "#f0f0f0" if i % 2 == 0 else "#e0e0e0"
        ax.axvspan(start_ts, end_ts, color=color, alpha=0.4, zorder=0)

    # Ticks com a data inicial de cada janela (dd/MM)
    tick_pos = [pd.Timestamp(w[0]) for w in WINDOWS]
    tick_labels = [
        f"{ts.strftime("%d/%m")}\nJan. {i+1}" for i, ts in enumerate(tick_pos)]
    ax.set_xticks(tick_pos, tick_labels, rotation=0)

    # ax.set_title("Índice de Sentimento Diário — BlueSky (faixas = janelas)")
    ax.set_ylabel("Proporção de publicações positivas")
    ax.set_xlabel("Janelas")
    ax.set_ylim(0, 1)
    ax.legend(title="Candidato", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def aggregate(df):
    """
    Calcula, por janela e candidato,
    - volume total,
    - proporção de posts positivos (sentiment_index)
    - share_pos  = pos_cand / Σ pos_todos
    - share_neg  = neg_cand / Σ neg_cand
    - share_tot  = tot_cand / Σ tot_cand
    """

    df["pos"] = (df["predicted_class"] == 1).astype(int)
    df["neg"] = (df["predicted_class"] == 0).astype(int)
    agg = (
        df.groupby(["window", "subject"])
        .agg(total_posts=("predicted_class", "size"),
             pos_posts=("pos", "sum"),
             neg_posts=("neg", "sum"))
        .reset_index()
    )

    agg["sentiment_index"] = agg["pos_posts"] / agg["total_posts"]

    # Shares por janela
    for col, new in [("pos_posts", "share_pos"),
                     ("neg_posts", "share_neg"),
                     ("total_posts", "share_tot")]:
        agg[new] = agg.groupby("window")[col].apply(
            lambda x: x / x.sum()).values
    return agg


def pearson_and_reg(agg, polls):
    """
    Calcula e loga a correlação e regressão entre share_pos/share_neg/share_tot e intenção de voto.
    """

    polls_idx = polls.reset_index().index + 1
    results = []
    for cand in CANDIDATES:
        col = POLL_COL[cand]
        poll_series = polls[col]
        for metric, tag in [("share_pos", "POS"),
                            ("share_neg", "NEG"),
                            ("share_tot", "TOT")]:
            df_c = (
                agg[agg["subject"] == cand]
                .merge(poll_series, left_on="window", right_on=polls_idx)
                .rename(columns={col: "vote"})
            )
            x = df_c[metric]
            y = df_c["vote"]
            r, p = pearsonr(x, y)
            X = sm.add_constant(x)
            beta = sm.OLS(y, X).fit()
            logging.info(f"{cand} [{tag}] | r={r:.3f} (p={p:.4f}) "
                         f"β1={beta.params[1]:.3f} R²={beta.rsquared:.3f}")
            results.append((cand, tag, r, p, beta.params[1], beta.rsquared))
    res_df = pd.DataFrame(results, columns=[
        "candidato", "métrica", "pearson_r", "p_val",
        "beta1", "r2"
    ])
    export_csv(res_df, "pearson_reg")

    return results


def plot_volume_by_window(agg, path):
    """Plota o volume total de postagens por janela/candidato."""
    plt.figure()
    sns.barplot(data=agg, x="window", y="total_posts",
                hue="subject", dodge=True)
    plt.legend(title="Candidato")
    # plt.title("Volume de postagens por janela e candidato")
    plt.ylabel("Número de publicações")
    plt.xlabel("Janelas")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_share_vs_vote_first_round(agg, polls, path):
    """Plota o percentual de postagens (share_pos) X intenção de voto."""
    first_win = agg[agg["window"] <= 7]
    first_polls = polls.iloc[:7]

    merged = (
        first_win
        .merge(
            first_polls.assign(window=first_polls.reset_index().index + 1),
            on="window"
        )
    )
    export_csv(merged, "share_vs_vote")

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, cand in enumerate(CANDIDATES):
        df_c = (
            first_win[first_win["subject"] == cand]
            .merge(
                first_polls[["date", POLL_COL[cand]]],
                left_on="window",
                right_on=first_polls.reset_index().index + 1
            )
        )

        axes[i].plot(df_c["window"], df_c["share_pos"] * 100,
                     label="% posts positivos", marker="o", color="#1f77b4")
        axes[i].plot(df_c["window"], df_c[POLL_COL[cand]],
                     label="% intenção de voto", marker="s", color="#ff7f0e")

        axes[i].set_title(cand)
        axes[i].set_xticks(df_c["window"])
        axes[i].legend(loc="upper right", fontsize=8)     # ← NOVO

    axes[-1].legend(loc="upper center", bbox_to_anchor=(-0.1, -0.25), ncol=2)
    # fig.suptitle("Posts positivos (%) × Intenção de voto (%) — 1º turno")
    fig.tight_layout(rect=[0, .03, 1, .95])
    fig.savefig(path)
    plt.close()


def export_csv(df: pd.DataFrame, name: str):
    """
    Salva o DataFrame como CSV.
    """
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    logging.info(f"CSV salvo em {path}")


def main():
    df, polls = load_data()
    df = assign_windows(df)
    plot_sentiment_series(df, OUT_DIR / "fig1_sent_series.png")

    agg = aggregate(df)
    export_csv(agg, "window_agg")
    pearson_and_reg(agg, polls)
    plot_volume_by_window(agg, OUT_DIR / "fig2_volume.png")
    plot_share_vs_vote_first_round(agg, polls,
                                   OUT_DIR / "fig3_share_vs_vote.png")

    logging.info("Pipeline concluído.")


if __name__ == "__main__":
    main()
