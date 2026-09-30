"""Generate a Portuguese scientific report and plots from completed experiments.

This module never selects hyperparameters or changes fitted models. Its paired
comparison is descriptive: repeated seeds are not a proof of equivariance or a
statistical significance test.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


MODEL_ORDER = ("ridge", "mlp", "mlp_augmented")
LABELS = {
    "ridge": "Ridge",
    "mlp": "MLP",
    "mlp_augmented": "MLP + rotações",
}
COLORS = {"ridge": "#667085", "mlp": "#1774B8", "mlp_augmented": "#D35B26"}
DOMAIN_LABELS = {"id": "Na distribuição (ID)", "ood": "Extrapolação (OOD)"}


def _number(value: Any) -> float | None:
    """Return finite numeric data; absent/nonfinite metrics remain explicit gaps."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stats(rows: list[dict], key: str) -> tuple[float, float, int]:
    values = [_number(row.get(key)) for row in rows]
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(finite):
        return float("nan"), float("nan"), 0
    return float(finite.mean()), float(finite.std(ddof=1)) if len(finite) > 1 else 0.0, len(finite)


def _format(number: Any) -> str:
    value = _number(number)
    if value is None:
        return "n/d"
    if value == 0:
        return "0"
    return f"{value:.4g}"


def _summary(rows: list[dict], key: str) -> str:
    mean, std, count = _stats(rows, key)
    if not count:
        return "n/d"
    if count == 1:
        summary = f"{_format(mean)} (1 semente)"
    else:
        summary = f"{_format(mean)} ± {_format(std)}"
    if count != len(rows):
        summary += f" [{count}/{len(rows)} finitas]"
    return summary


def _group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return dict(groups)


def _decorate(axis: plt.Axes, title: str, ylabel: str, *, log: bool = True) -> None:
    axis.set_title(title, fontsize=11)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22, which="major")
    axis.spines[["top", "right"]].set_visible(False)
    if log:
        axis.set_yscale("log")


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout(pad=1.3)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_training(output_dir: Path, largest_size: int) -> None:
    histories = []
    for path in sorted((output_dir / "histories").glob("*.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        if int(history.get("train_size", -1)) == largest_size and history.get("history"):
            histories.append(history)
    seeds = sorted({item["seed"] for item in histories})
    first_seed = seeds[0] if seeds else None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    plotted = 0
    for item in histories:
        if item["seed"] != first_seed or item["model"] not in ("mlp", "mlp_augmented"):
            continue
        model = item["model"]
        epochs = [row["epoch"] for row in item["history"]]
        for axis, key in zip(axes, ("train_loss", "val_loss")):
            losses = [_number(row.get(key)) for row in item["history"]]
            values = [max(value, 1e-12) if value is not None else np.nan for value in losses]
            axis.plot(epochs, values, color=COLORS[model], label=LABELS[model], linewidth=1.8)
            if _number(item.get("best_epoch")) is not None:
                axis.axvline(item["best_epoch"], color=COLORS[model], linestyle=":", alpha=0.6)
        plotted += 1
    for axis, title in zip(axes, ("Treino", "Validação")):
        _decorate(axis, title, "Perda relativa quadrática média")
        axis.set_xlabel("Época")
        if plotted:
            axis.legend(fontsize=9)
        else:
            axis.text(0.5, 0.5, "Históricos indisponíveis", transform=axis.transAxes, ha="center")
    caption = f"n = {largest_size}; semente = {first_seed}" if seeds else f"n = {largest_size}"
    figure.suptitle(f"Curvas de otimização ({caption})", fontsize=12)
    _save(figure, output_dir / "figures" / "training_curves.png")


def _plot_learning(output_dir: Path, records: list[dict]) -> None:
    grouped = _group(records, "model", "train_size", "domain")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, domain in enumerate(("id", "ood")):
        for row_index, (key, statistic) in enumerate((("error_median", "Mediana"), ("error_p95", "Percentil 95"))):
            axis = axes[row_index, column]
            for model in MODEL_ORDER:
                sizes = sorted({size for m, size, d in grouped if m == model and d == domain})
                means, stds = [], []
                for size in sizes:
                    mean, std, _ = _stats(grouped[model, size, domain], key)
                    means.append(mean)
                    stds.append(std)
                if sizes:
                    average, spread = np.asarray(means), np.asarray(stds)
                    axis.plot(sizes, np.maximum(average, 1e-12), "o-", label=LABELS[model], color=COLORS[model])
                    axis.fill_between(sizes, np.maximum(average - spread, 1e-12), np.maximum(average + spread, 1e-12), color=COLORS[model], alpha=0.14)
            _decorate(axis, f"{DOMAIN_LABELS[domain]}: {statistic.lower()}", f"{statistic} do erro relativo")
            axis.set_xlabel("Número de exemplos fonte de treino")
            axis.set_xscale("log")
            all_sizes = sorted({size for _, size, _ in grouped})
            axis.set_xticks(all_sizes, [str(size) for size in all_sizes])
            axis.xaxis.set_minor_locator(NullLocator())
            if axis.lines:
                axis.legend(fontsize=9)
    figure.suptitle("Curvas de aprendizado: média ± desvio padrão entre sementes", fontsize=12)
    _save(figure, output_dir / "figures" / "learning_curves.png")


def _points_by_model(axis: plt.Axes, rows: list[dict], key: str) -> None:
    positions, labels = [], []
    for position, model in enumerate(MODEL_ORDER):
        model_rows = [row for row in rows if row["model"] == model]
        mean, std, count = _stats(model_rows, key)
        if not count:
            continue
        displayed = max(mean, 1e-12)
        lower_error = displayed - max(mean - std, 1e-12)
        axis.errorbar(position, displayed, yerr=[[lower_error], [std]], fmt="o", capsize=5, color=COLORS[model], markersize=6)
        individual = [_number(row.get(key)) for row in model_rows]
        axis.scatter([position] * len(individual), [max(value, 1e-12) if value is not None else np.nan for value in individual], s=12, color=COLORS[model], alpha=0.4)
        positions.append(position)
        labels.append(LABELS[model])
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.45, 2.45)


def _plot_equivariance(output_dir: Path, records: list[dict], largest_size: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.7))
    for column, domain in enumerate(("id", "ood")):
        rows = [row for row in records if row["domain"] == domain and row["train_size"] == largest_size]
        for row_index, (key, group_name) in enumerate((("eq_so7_median", "Rotações SO(7)"), ("eq_gl7_median", "Mudanças de base GL⁺(7)"))):
            axis = axes[row_index, column]
            _points_by_model(axis, rows, key)
            _decorate(axis, f"{DOMAIN_LABELS[domain]}: {group_name}", "Mediana do defeito relativo")
    figure.suptitle(f"Equivariância: média ± desvio padrão entre sementes (n = {largest_size})", fontsize=12)
    _save(figure, output_dir / "figures" / "equivariance.png")


def _scale_key(row: dict, scale: float, suffix: str) -> str | None:
    """Accommodate JSON keys formatted as either '2' or '2.0'."""
    for key in row:
        if not key.startswith("scale_") or not key.endswith(f"_{suffix}"):
            continue
        numeric = key[len("scale_") : -len(f"_{suffix}")]
        try:
            if math.isclose(float(numeric), scale, rel_tol=1e-10, abs_tol=0.0):
                return key
        except ValueError:
            pass
    return None


def _plot_scaling(output_dir: Path, records: list[dict], largest_size: int, config: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.7))
    scales = sorted(float(value) for value in config.get("scale_factors", []) if float(value) > 0)
    any_individual = False
    for column, domain in enumerate(("id", "ood")):
        rows = [row for row in records if row["domain"] == domain and row["train_size"] == largest_size]
        for row_index, (suffix, aggregate, title) in enumerate((("defect_median", "scale_defect_median", "Defeito da lei de escala"), ("error_median", "scale_error_median", "Erro contra a métrica exata escalada"))):
            axis = axes[row_index, column]
            individual_available = any(_scale_key(row, scale, suffix) is not None for row in rows for scale in scales)
            any_individual = any_individual or individual_available
            if individual_available:
                for model in MODEL_ORDER:
                    model_rows = [row for row in rows if row["model"] == model]
                    means, stds = [], []
                    for scale in scales:
                        values = [{"value": row.get(_scale_key(row, scale, suffix))} for row in model_rows]
                        mean, std, _ = _stats(values, "value")
                        means.append(mean)
                        stds.append(std)
                    means_array, stds_array = np.asarray(means), np.asarray(stds)
                    if np.isfinite(means_array).any():
                        axis.plot(scales, np.maximum(means_array, 1e-12), "o-", color=COLORS[model], label=LABELS[model])
                        axis.fill_between(scales, np.maximum(means_array - stds_array, 1e-12), np.maximum(means_array + stds_array, 1e-12), alpha=0.14, color=COLORS[model])
                axis.set_xscale("log")
                axis.set_xticks(scales, [f"{scale:g}" for scale in scales])
                axis.xaxis.set_minor_locator(NullLocator())
                axis.set_xlabel("Fator t aplicado à forma φ")
                if axis.lines:
                    axis.legend(fontsize=9)
            else:
                _points_by_model(axis, rows, aggregate)
            _decorate(axis, f"{DOMAIN_LABELS[domain]}: {title}", "Mediana relativa")
    detail = "por fator" if any_individual else "agregada nos fatores disponíveis"
    figure.suptitle(f"Homogeneidade F(tφ) = t²ᐟ³F(φ), {detail} (n = {largest_size})", fontsize=12)
    _save(figure, output_dir / "figures" / "scaling.png")


def _paired_assessment(records: list[dict], largest_size: int, config: dict) -> list[str]:
    """Apply a predeclared descriptive success criterion to paired seeds."""
    tolerance = float(config.get("accuracy_tolerance_relative", 0.05))
    relevant = [row for row in records if row["train_size"] == largest_size and row["model"] in ("mlp", "mlp_augmented")]
    lookup = {(row["model"], row["seed"], row["domain"]): row for row in relevant}
    seeds = list(config.get("seeds", sorted({row["seed"] for row in relevant})))
    lines = [
        "## Comparação pareada da hipótese",
        "",
        f"Critério operacional declarado na configuração: no maior tamanho de treino ({largest_size}), "
        f"a MLP com rotações deve reduzir a mediana do defeito SO(7) em cada semente e em ambos os conjuntos; "
        f"o aumento relativo da mediana do erro preditivo deve ser no máximo {100 * tolerance:g}% em ID e OOD, também por semente. "
        "A tolerância quantifica 'sem perda material de acurácia'; não é uma significância estatística. "
        "O diagnóstico GL⁺(7) é relatado separadamente.",
        "",
        "| Semente | Domínio | Variação do erro mediano | Variação do defeito SO(7) | Acurácia aceitável | Defeito menor |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    all_pass, complete = bool(seeds), bool(seeds)
    for seed in seeds:
        for domain in ("id", "ood"):
            baseline, augmented = lookup.get(("mlp", seed, domain)), lookup.get(("mlp_augmented", seed, domain))
            if baseline is None or augmented is None:
                complete = False
                lines.append(f"| {seed} | {domain.upper()} | n/d | n/d | Par ausente | Par ausente |")
                continue
            errors = [_number(row.get("error_median")) for row in (baseline, augmented)]
            defects = [_number(row.get("eq_so7_median")) for row in (baseline, augmented)]
            if None in errors or None in defects:
                complete = False
                lines.append(f"| {seed} | {domain.upper()} | n/d | n/d | Métrica ausente | Métrica ausente |")
                continue
            error_base, error_aug = errors
            defect_base, defect_aug = defects
            accuracy_pass = error_aug <= (1 + tolerance) * error_base
            equivariance_pass = defect_aug < defect_base
            all_pass = all_pass and accuracy_pass and equivariance_pass
            def change_label(base: float, aug: float) -> str:
                if base == 0:
                    return "0%" if aug == 0 else "base zero; aumento"
                return f"{100 * (aug / base - 1):+.2f}%"
            lines.append(f"| {seed} | {domain.upper()} | {change_label(error_base, error_aug)} | {change_label(defect_base, defect_aug)} | {'Sim' if accuracy_pass else 'Não'} | {'Sim' if equivariance_pass else 'Não'} |")
    lines.append("")
    if not complete:
        lines.append("**Conclusão: comparação inconclusiva.** Faltam pares ou métricas finitas para aplicar o critério em todas as sementes configuradas.")
    elif len(seeds) < 3:
        status = "satisfeito" if all_pass else "não satisfeito"
        lines.append(f"**Conclusão: execução de verificação com menos de três sementes.** O critério foi {status} nos pares disponíveis, mas a repetição por três sementes prevista no planejamento ainda não foi cumprida por esta execução.")
    elif all_pass:
        lines.append("**Conclusão: o critério experimental foi satisfeito nesta configuração.** Os pares observados apoiam a hipótese de redução do defeito de rotação sem perda de acurácia acima da tolerância fixada. Isso não estabelece uma garantia de equivariância, significância estatística ou comportamento fora das distribuições avaliadas.")
    else:
        lines.append("**Conclusão: o critério experimental não foi satisfeito nesta configuração.** Pelo menos uma semente/domínio não reduziu o defeito SO(7) ou ultrapassou a tolerância de erro preditivo. Trata-se de um resultado negativo ou misto para esta configuração; não demonstra impossibilidade de benefício em outros orçamentos.")
    lines.extend(["", "A comparação usa sementes finitas e dois conjuntos finitos de teste; ela não constitui demonstração matemática nem teste de hipótese com controle de significância. Os resultados de teste não são usados para escolher checkpoints ou hiperparâmetros."])
    return lines


def _metadata_lines(metadata: dict) -> list[str]:
    lines = ["## Ambiente e rastreabilidade", "", "Os dados abaixo registram o ambiente e as verificações executadas; valores ausentes não são inferidos.", ""]
    for key, title in (("versions", "Versões"), ("platform", "Plataforma"), ("data_validation", "Validação matemática dos dados"), ("git", "Estado Git")):
        if key in metadata:
            lines.extend([f"### {title}", "", "```json", json.dumps(metadata[key], ensure_ascii=False, indent=2, default=str), "```", ""])
    return lines


def build_report(output_dir: Path, records: list[dict], config: dict, metadata: dict) -> None:
    """Save four PNG figures and ``report.md`` from already measured records.

    Each record describes one (model, seed, train_size, domain) combination.
    Medians and percentiles are computed upstream per seed. The report averages
    those statistics over seeds, rather than claiming a pooled population median.
    """
    output_dir = Path(output_dir)
    if not records:
        raise ValueError("Cannot build a scientific report without evaluation records.")
    identities = [(row["model"], row["seed"], row["train_size"], row["domain"]) for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate (model, seed, train_size, domain) evaluation records.")
    largest_size = max(int(row["train_size"]) for row in records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "normal"}):
        _plot_training(output_dir, largest_size)
        _plot_learning(output_dir, records)
        _plot_equivariance(output_dir, records, largest_size)
        _plot_scaling(output_dir, records, largest_size, config)

    grouped = _group(records, "model", "train_size", "domain")
    seeds = sorted({row["seed"] for row in records})
    sizes = sorted({row["train_size"] for row in records})
    lines = [
        "# Aprendizado da métrica induzida por uma estrutura G₂",
        "",
        "## Pergunta e protocolo",
        "",
        "Este experimento compara ridge, uma MLP e a mesma MLP com aumento de dados por rotações, "
        "para aproximar a aplicação algébrica pontual de uma 3-forma positiva aos coeficientes de sua métrica. "
        "A fórmula exata é conhecida e serve de referência; o objetivo é investigar generalização e equivariância aprendida.",
        "",
        f"Configuração: **{config.get('name', 'sem nome')}**. Sementes observadas: **{', '.join(map(str, seeds))}**. "
        f"Tamanhos de treino observados: **{', '.join(map(str, sizes))}**. "
        f"Orçamento máximo configurado: **{config.get('max_epochs', 'n/d')} épocas** por MLP, "
        "com parada antecipada pela validação.",
        "",
        "ID designa exemplos independentes na faixa de geração do treino; OOD designa exemplos "
        "com pelo menos um logaritmo de valor singular fora dessa faixa. As transformações de teste "
        "são novas. A ação de SO(7) usada para aumentar os dados preserva os valores singulares. "
        "Os pares fonte são separados antes do aumento de dados; a padronização usa somente o treino.",
        "",
        "A MLP recebe 35 coeficientes da forma e produz 28 parâmetros de um fator triangular, "
        "com diagonal positiva, cuja matriz de Gram é a previsão da métrica. Essa parametrização "
        "impõe positividade em aritmética exata, mas não equivariância. Ridge prevê as entradas "
        "da matriz simétrica sem impor positividade.",
        "",
        "## Erros preditivos",
        "",
        "O erro de cada exemplo é `||g_pred - g||_F / ||g||_F`. Cada célula agrega a estatística "
        "calculada em cada semente: média ± desvio padrão amostral entre sementes. Portanto, "
        "a média das medianas não é a mediana de todos os exemplos reunidos. Uma única semente "
        "não estima a variação entre inicializações. `n/d` indica métrica ausente ou não finita.",
        "",
        "| Modelo | Treino | Domínio | Erro mediano | Percentil 95 | Erro médio |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for size in sizes:
        for domain in ("id", "ood"):
            for model in MODEL_ORDER:
                rows = grouped.get((model, size, domain), [])
                if rows:
                    lines.append(f"| {LABELS[model]} | {size} | {domain.upper()} | {_summary(rows, 'error_median')} | {_summary(rows, 'error_p95')} | {_summary(rows, 'error_mean')} |")
    lines.extend([
        "", "![Curvas de aprendizado](figures/learning_curves.png)", "",
        "As faixas nos gráficos mostram ± um desvio padrão entre sementes, não intervalos de confiança. "
        "Para legibilidade em escala logarítmica, valores plotados abaixo de `1e-12` são limitados "
        "a esse piso; as tabelas mantêm os valores numéricos originais.",
        "", "## Equivariância e homogeneidade", "",
        "O defeito de equivariância compara `g_pred(C*φ)` com `Cᵀ g_pred(φ) C`, normalizado "
        "por `||Cᵀ g C||_F`. A normalização usa a resposta exata. O defeito de escala "
        "compara `g_pred(tφ)` com `t^(2/3) g_pred(φ)`; o erro escalado compara a primeira "
        "quantidade à resposta exata `t^(2/3) g`. Um modelo pode ter baixo defeito e ainda "
        "prever a métrica errada, por isso os dois tipos de medida são separados.",
        "", f"Resultados no maior conjunto fonte de treino: **{largest_size}** exemplos.", "",
        "| Modelo | Domínio | SO(7): mediana | SO(7): p95 | GL⁺(7): mediana | GL⁺(7): p95 | Defeito de escala | Erro escalado |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for domain in ("id", "ood"):
        for model in MODEL_ORDER:
            rows = grouped.get((model, largest_size, domain), [])
            if rows:
                metrics = [_summary(rows, key) for key in ("eq_so7_median", "eq_so7_p95", "eq_gl7_median", "eq_gl7_p95", "scale_defect_median", "scale_error_median")]
                lines.append(f"| {LABELS[model]} | {domain.upper()} | " + " | ".join(metrics) + " |")
    lines.extend(["", "![Defeitos de equivariância](figures/equivariance.png)", "", "![Defeito e erro de escala](figures/scaling.png)", ""])
    lines.extend(_paired_assessment(records, largest_size, config))
    lines.extend(["", "## Positividade e custo de treinamento", "",
        "As falhas de positividade são contadas nas previsões avaliadas; o ridge não é corrigido "
        "por projeção no cone positivo. Os tempos refletem este ambiente e incluem apenas "
        "as etapas medidas pelo treinador; não estabelecem vantagem de velocidade sobre a fórmula exata.", "",
        "| Modelo | Domínio | Falhas de SPD / avaliações, somadas nas sementes | Tempo de treino (s), média ± DP | Melhor época, média ± DP |",
        "| --- | --- | ---: | ---: | ---: |"])
    for domain in ("id", "ood"):
        for model in MODEL_ORDER:
            rows = grouped.get((model, largest_size, domain), [])
            if rows:
                failures = sum(int(row.get("spd_failures", 0)) for row in rows)
                total = sum(int(row.get("n", 0)) for row in rows)
                epoch = "não se aplica" if model == "ridge" else _summary(rows, "best_epoch")
                lines.append(f"| {LABELS[model]} | {domain.upper()} | {failures} / {total} | {_summary(rows, 'training_seconds')} | {epoch} |")
    lines.extend(["", "O tempo de treino reaparece nas linhas ID/OOD porque o mesmo modelo é avaliado nos dois conjuntos; ele não deve ser somado duas vezes.", "",
        "## Curvas de otimização", "", "![Curvas de treino e validação](figures/training_curves.png)", "",
        "O gráfico mostra a primeira semente disponível no maior tamanho de treino. As linhas "
        "verticais pontilhadas marcam as épocas selecionadas pela validação. Os históricos "
        "completos das demais sementes permanecem em `histories/`. A perda é relativa e "
        "quadrática, ao passo que os erros das tabelas não são elevados ao quadrado.", "",
        "## Limitações e interpretação", "",
        "- A distribuição fonte já é invariante sob rotações, porque o fator ortogonal direito "
        "é Haar. Augmentation explora novas versões de uma amostra finita; não alarga o suporte populacional.",
        "- Equivariância por rotações, naturalidade por mudanças de base gerais e homogeneidade "
        "são diagnósticos distintos. Nenhum é garantido pela MLP genérica.",
        "- O orçamento máximo é igual entre MLPs, mas a parada antecipada pode produzir "
        "números efetivos de épocas diferentes. Os históricos permitem inspecionar essa diferença.",
        "- Uma amostra finita não verifica identidades para todas as formas positivas. Resultados "
        "OOD se referem à faixa configurada e não a extrapolação arbitrária.",
        "- Este projeto não resolve EDPs, não impõe ausência de torção e não descobre uma "
        "nova construção da métrica. A referência é uma identidade matemática conhecida.",
        "- A assistência de IA na implementação deve ser documentada junto dos testes independentes; "
        "a responsabilidade pelas afirmações matemáticas permanece na análise e validação dos resultados.", ""])
    lines.extend(_metadata_lines(metadata))
    lines.extend(["## Configuração executada", "", "```json", json.dumps(config, ensure_ascii=False, indent=2, default=str), "```", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
