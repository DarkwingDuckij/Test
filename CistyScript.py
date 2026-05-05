import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ── Konfigurace ────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)

TOP_5_LIGY = ["Premier League", "LaLiga", "Serie A", "1.Bundesliga", "Ligue 1"]

DTYPE_MAPPER = {
    "Jméno":               "string",
    "Pozice":              "string",
    "Věk":                 "Int64",
    "Původní tým":         "string",
    "Původní liga":        "string",
    "Nový tým":            "string",
    "Nová  Liga":          "string",   # Extra mezera v originálním CSV
    "Sezóna":              "string",
    "Odhadovaná hodnota":  "Float64",
    "Přestupová částka":   "Float64",
}


# ── 1. Načtení dat ─────────────────────────────────────────────────────────────

def nacti_data(filepath: str) -> pd.DataFrame:
    """Načte CSV se správnými datovými typy a provede základní normalizaci."""
    df = pd.read_csv(filepath, dtype=DTYPE_MAPPER)

    # Věk == 0 je chybný záznam
    df.loc[df["Věk"] == 0, "Věk"] = pd.NA

    # Normalizace názvu sloupce
    df.rename(columns={"Nová  Liga": "Nová Liga"}, inplace=True)

    return df


# ── 2. Čistění a transformace ──────────────────────────────────────────────────

def vycisti_a_transformuj(df: pd.DataFrame) -> pd.DataFrame:
    """Odstraní nadbytečné mezery v textových sloupcích a přidá odvozené proměnné."""
    # Odstranění mezer z textových sloupců
    for col in df.columns:
        if df[col].dtype == "string":
            df[col] = df[col].str.strip()

    # Odvozené sloupce
    df["Sezóna začátek"] = df["Sezóna"].str.extract(r"(\d{4})").astype("Int64")
    df["V rámci stejné ligy"] = df["Původní liga"] == df["Nová Liga"]
    df["Odhadovaná hodnota dostupná"] = df["Odhadovaná hodnota"].notna()
    df["Poměr částka / hodnota"] = df["Přestupová částka"] / df["Odhadovaná hodnota"]
    df["Rozdíl částka - hodnota"] = df["Přestupová částka"] - df["Odhadovaná hodnota"]
    df["Věková skupina"] = pd.cut(
        df["Věk"],
        bins=[0, 20, 23, 26, 29, 40],
        labels=["<=20", "21-23", "24-26", "27-29", "30+"],
    )

    return df


# ── 3. Pomocná funkce pro uložení grafů ───────────────────────────────────────

def uloz(fig: plt.Figure, output_dir: Path, nazev: str) -> None:
    cesta = output_dir / f"{nazev}.png"
    fig.savefig(cesta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {cesta.name}")


# ── 4. Grafy ───────────────────────────────────────────────────────────────────

def graf_vyvoj_trhu(df: pd.DataFrame, output_dir: Path) -> None:
    """4.1 Vývoj trhu v čase – celková částka (sloupce) + průměrná cena (linie)."""
    sezony = (
        df.groupby("Sezóna začátek")
        .agg(
            Celkem_castka=("Přestupová částka", "sum"),
            Prumer_castka=("Přestupová částka", "mean"),
        )
        .reset_index()
    )

    x_pozice = range(len(sezony))
    x_popisky = sezony["Sezóna začátek"].astype(str)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(x_pozice, sezony["Celkem_castka"], color="#2a9d8f", alpha=0.85, label="Celková částka")
    ax1.set_title("Vývoj trhu v čase")
    ax1.set_xlabel("Začátek sezony")
    ax1.set_ylabel("Celková částka přestupů")
    ax1.set_xticks(list(x_pozice))
    ax1.set_xticklabels(x_popisky, rotation=45)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000_000:.1f} mld."))

    ax2 = ax1.twinx()
    ax2.plot(x_pozice, sezony["Prumer_castka"], marker="o", color="#e76f51",
             linewidth=2.5, label="Průměrná částka")
    ax2.set_ylabel("Průměrná částka na transfer")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f} mil."))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    uloz(fig, output_dir, "01_vyvoj_trhu")


def graf_bilance_lig(df: pd.DataFrame, output_dir: Path) -> None:
    """4.2 Kdo je kupec a kdo exportér – top 10 lig podle celkového objemu."""
    bilance_lig = (
        pd.concat(
            [
                df.groupby("Nová Liga")["Přestupová částka"].sum().rename("Nakoupeno"),
                df.groupby("Původní liga")["Přestupová částka"].sum().rename("Prodano"),
            ],
            axis=1,
        )
        .fillna(0)
        .assign(
            Cista_bilance=lambda x: x["Nakoupeno"] - x["Prodano"],
            Celkovy_objem=lambda x: x["Nakoupeno"] + x["Prodano"],
        )
        .sort_values("Cista_bilance", ascending=False)
        .reset_index()
        .rename(columns={"index": "Liga"})
    )

    graf_df = (
        bilance_lig.sort_values("Celkovy_objem", ascending=False)
        .head(10)
        .sort_values("Celkovy_objem", ascending=True)
    )
    graf_df_long = graf_df.melt(
        id_vars="Liga",
        value_vars=["Nakoupeno", "Prodano"],
        var_name="Typ toku",
        value_name="Castka",
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=graf_df_long,
        y="Liga",
        x="Castka",
        hue="Typ toku",
        palette={"Nakoupeno": "#2a9d8f", "Prodano": "#e76f51"},
        ax=ax,
    )
    ax.set_title("Nákupy vs. prodeje podle lig (top 10 podle objemu)")
    ax.set_xlabel("Přestupová částka")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1_000_000_000:.1f} mld."))
    ax.legend(title="")
    plt.tight_layout()
    uloz(fig, output_dir, "02_bilance_lig")


def graf_top_kluby(df: pd.DataFrame, output_dir: Path) -> None:
    """4.3 Top 10 nakupujících a prodávajících klubů."""
    top_nakup_mil  = df.groupby("Nový tým")["Přestupová částka"].sum().nlargest(10).sort_values() / 1_000_000
    top_prodej_mil = df.groupby("Původní tým")["Přestupová částka"].sum().nlargest(10).sort_values() / 1_000_000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(top_nakup_mil.index,  top_nakup_mil.values,  color="#2a9d8f")
    axes[0].set_title("Top 10 kupujících klubů (2000–2019)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Celková útrata (mil. EUR)")
    axes[1].barh(top_prodej_mil.index, top_prodej_mil.values, color="#e76f51")
    axes[1].set_title("Top 10 prodávajících klubů (2000–2019)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Celkové příjmy (mil. EUR)")
    plt.tight_layout()
    uloz(fig, output_dir, "03_top_kluby")


def graf_trzni_hodnota(df: pd.DataFrame, output_dir: Path) -> None:
    """4.4 Tržní hodnota vs. skutečná přestupová cena."""
    graf_df = df[df["Odhadovaná hodnota dostupná"]].copy()
    graf_df["Odhadovaná hodnota (mil. EUR)"] = graf_df["Odhadovaná hodnota"] / 1_000_000
    graf_df["Přestupová částka (mil. EUR)"]  = graf_df["Přestupová částka"]  / 1_000_000
    graf_df["Poměr oříznutý"] = graf_df["Poměr částka / hodnota"].clip(upper=4)

    median_pomer = graf_df["Poměr částka / hodnota"].median()
    max_osa = max(
        graf_df["Odhadovaná hodnota (mil. EUR)"].max(),
        graf_df["Přestupová částka (mil. EUR)"].max(),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(
        data=graf_df,
        x="Odhadovaná hodnota (mil. EUR)",
        y="Přestupová částka (mil. EUR)",
        alpha=0.25, s=22, color="#e78ac3", edgecolor=None, ax=ax1,
    )
    ax1.plot([0, max_osa], [0, max_osa], linestyle="--", color="red", label="Cena = hodnota")
    ax1.set_title("Tržní hodnota vs. skutečná cena")
    ax1.set_xlabel("Tržní hodnota (mil. EUR)")
    ax1.set_ylabel("Přestupová částka (mil. EUR)")
    ax1.legend()

    sns.histplot(data=graf_df, x="Poměr oříznutý", bins=35, color="#6ad1c0", ax=ax2)
    ax2.axvline(1, linestyle="--", color="red", label="Cena = hodnota")
    ax2.axvline(median_pomer, color="#e76f51", label=f"Median = {median_pomer:.2f}")
    ax2.set_title("Histogram: cena / trzni hodnota")
    ax2.set_xlabel("Pomer")
    ax2.set_ylabel("Pocet prestupu")
    ax2.legend()

    fig.suptitle("Vztah tržní hodnoty a přestupové ceny", fontsize=16, fontweight="bold")
    plt.tight_layout()
    uloz(fig, output_dir, "04_trzni_hodnota")


def graf_premie_podle_veku(df: pd.DataFrame, output_dir: Path) -> None:
    """4.5 Prémie nad tržní hodnotu dle věkové skupiny."""
    premie_podle_veku = (
        df[df["Odhadovaná hodnota dostupná"]]
        .groupby("Věková skupina", observed=False)
        .agg(
            Pocet_prestupu=("Jméno", "count"),
            Median_pomer=("Poměr částka / hodnota", "median"),
            Median_rozdil=("Rozdíl částka - hodnota", "median"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=premie_podle_veku, x="Věková skupina", y="Median_pomer",
                color="#287271", ax=ax)
    ax.set_title("Medián poměru částka / hodnota podle věku")
    ax.set_xlabel("Věková skupina")
    ax.set_ylabel("Medián poměru")
    plt.tight_layout()
    uloz(fig, output_dir, "05_premie_vek")


def graf_pozice(df: pd.DataFrame, output_dir: Path) -> None:
    """4.6 Distribuce přestupových částek podle pozice (boxplot)."""
    pocet_podle_pozice = df["Pozice"].value_counts()
    vybrane_pozice = pocet_podle_pozice[pocet_podle_pozice >= 80].index

    graf_df = df[df["Pozice"].isin(vybrane_pozice)].copy()
    graf_df["Přestupová částka (mil. EUR)"] = graf_df["Přestupová částka"] / 1_000_000

    poradi = (
        graf_df.groupby("Pozice")["Přestupová částka (mil. EUR)"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=graf_df,
        y="Pozice",
        x="Přestupová částka (mil. EUR)",
        order=poradi,
        showfliers=False,
        color="#5aa9a2",
        ax=ax,
    )
    ax.set_title("Distribuce přestupových částek podle pozice")
    ax.set_xlabel("Přestupová částka (mil. EUR)")
    ax.set_ylabel("")
    plt.tight_layout()
    uloz(fig, output_dir, "06_pozice")


def graf_heatmap_ligy(df: pd.DataFrame, output_dir: Path) -> None:
    """4.7 Heatmapa přestupů mezi top-5 ligami."""
    heatmap_top5 = (
        df[df["Původní liga"].isin(TOP_5_LIGY) & df["Nová Liga"].isin(TOP_5_LIGY)]
        .groupby(["Původní liga", "Nová Liga"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=TOP_5_LIGY, columns=TOP_5_LIGY, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        heatmap_top5, annot=True, fmt="g", cmap="Blues",
        cbar_kws={"label": "Počet přestupů"}, ax=ax,
    )
    ax.set_title("Počet přestupů mezi top-5 ligami")
    ax.set_xlabel("Cílová liga")
    ax.set_ylabel("Zdrojová liga")
    plt.tight_layout()
    uloz(fig, output_dir, "07_heatmap_ligy")


# ── 5. Hlavní funkce ───────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Načítám data...")
    df = nacti_data(input_path)
    df = vycisti_a_transformuj(df)
    print(f"  -> {len(df)} záznamů připraveno")

    print("Generuji grafy...")
    graf_vyvoj_trhu(df, out)
    graf_bilance_lig(df, out)
    graf_top_kluby(df, out)
    graf_trzni_hodnota(df, out)
    graf_premie_podle_veku(df, out)
    graf_pozice(df, out)
    graf_heatmap_ligy(df, out)

    print("Hotovo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA – Fotbalové přestupy 2000–2019")
    p.add_argument("--input",      default="fotbal_prestupy_2000_2019.csv")
    p.add_argument("--output_dir", default="./output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir)
