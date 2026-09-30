# AI-Project — Métrica induzida por uma estrutura G₂

Projeto da disciplina **MM845 — uso de IA na pesquisa matemática**. Compara regressão ridge, uma MLP em PyTorch e a mesma MLP com augmentação por rotações para aproximar a aplicação `φ ↦ g_φ` em um espaço vetorial orientado de dimensão 7.

O planejamento é o arquivo [Planajamento de projeto.pdf](material-consulta/Planajamento%20de%20projeto.pdf). A rede recebe **35 coeficientes de uma 3-forma positiva** e retorna uma matriz simétrica 7×7 positiva definida. Trata-se de um benchmark algébrico com fórmula exata conhecida.

## Instalação

Python 3.10 ou superior. O ambiente Conda é a opção recomendada porque reúne as
dependências do projeto, dos testes, da leitura de PDFs e do notebook. No PowerShell,
dentro da pasta do projeto:

```powershell
conda env create --file environment.yml
conda activate g2metric-mm845
python -m pytest -q
```

O arquivo `environment.yml` configura Python 3.12 e PyTorch para CPU, coerente com
`configs/full.json`. Ele também instala o próprio pacote `g2metric` em modo editável:
alterações feitas nos arquivos `.py` passam a valer sem reinstalação. Para atualizar
um ambiente já criado depois de mudanças no arquivo:

```powershell
conda env update --name g2metric-mm845 --file environment.yml --prune
conda activate g2metric-mm845
```

Para conferir o ambiente ativo ou removê-lo posteriormente:

```powershell
conda env list
conda deactivate
conda env remove --name g2metric-mm845
```

Se quiser usar CUDA, instale a distribuição PyTorch compatível com sua GPU e remova
`cpuonly` de uma cópia do arquivo antes de criar o ambiente. A opção abaixo usa
`venv` e continua disponível para quem não utiliza Conda:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O ambiente `.venv` preparado nesta sessão já contém as dependências. Não é necessário ativá-lo. Em Linux/macOS, substitua `.\.venv\Scripts\python.exe` por `.venv/bin/python`.

Para instalar especificamente a distribuição PyTorch para CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` registra as versões exatas usadas localmente. Inclui pacotes de leitura de PDF e o build CPU de PyTorch. Para reproduzir esse ambiente com `pip`, use o índice CPU com `--extra-index-url https://download.pytorch.org/whl/cpu`; o arquivo geral de requisitos é preferível em outras plataformas. O `environment.yml` declara faixas compatíveis em vez de reproduzir byte a byte o ambiente local.

## Executar

```powershell
# Testes matemáticos, de dados, modelos e integração
.\.venv\Scripts\python.exe -m pytest -q

# Verificação rápida de todo o fluxo (não é resultado científico)
.\.venv\Scripts\python.exe -m g2metric run --config configs/smoke.json --output outputs/minha-verificacao

# Experimento do planejamento: 30.000 pares, três seeds e três tamanhos de treino
.\.venv\Scripts\python.exe -m g2metric run --config configs/full.json --output outputs/minha-execucao

# Opcional: gerar somente os dados e seu manifesto
.\.venv\Scripts\python.exe -m g2metric generate --config configs/full.json --output data/generated/g2.npz
```

Cada execução requer uma pasta de saída nova ou vazia, para preservar resultados anteriores. O perfil completo executa 27 ajustes (9 ridge e 18 MLP), com no máximo 100 épocas por rede. O tempo depende do equipamento; CPU é o padrão. Para CUDA, instale o build compatível do PyTorch e altere `device` na configuração.

## Resultados

A execução completa já está disponível: [relatório e gráficos](outputs/full/report.md) e [interpretação dos resultados](docs/resultados.md). Foram concluídos os 27 ajustes, com 24 testes automatizados aprovados.

No maior conjunto de treino, a média das medianas do erro relativo nas três sementes foi:

| Modelo | Teste usual (ID) | Extrapolação (OOD) |
| --- | ---: | ---: |
| Ridge | 36,58% | 62,29% |
| MLP | 25,81% | 43,57% |
| MLP com rotações | 24,99% | 42,79% |

A augmentação melhorou a precisão e reduziu o defeito SO(7) nas três sementes, mas os erros ainda são elevados, especialmente fora da distribuição. Isso apoia a hipótese experimental nesta configuração; não demonstra equivariância exata nem uma aproximação de alta precisão.

A execução produz:

- `report.md`: comparação, variabilidade entre seeds e interpretação da hipótese;
- `metrics.csv` / `metrics.json`: erros mediano e p95, defeitos de equivariância, escala e falhas de positividade;
- `figures/`: curvas de treino/validação, curvas por tamanho de treino, equivariância e escala;
- `dataset.npz` / `dataset.json`: dados float64, IDs, splits, configuração e verificação da fórmula exata;
- `checkpoints/`: modelos e estatísticas de padronização, carregáveis para inferência;
- `histories/` e `metadata.json`: seleção por validação, versões, hashes do código/dados e orçamento efetivo;
- `predictions/`: resultados individuais dos testes e índices dos diagnósticos geométricos.

Toda a pasta `outputs/` é ignorada pelo Git, incluindo dados, pesos, predições, configurações copiadas de execuções, históricos, métricas, relatórios automáticos e figuras. Esses arquivos continuam disponíveis localmente, mas não acompanham um clone do repositório. `data/generated/`, arquivos binários de dados/modelos e logs também são ignorados. Os códigos-fonte, configurações de entrada em `configs/`, documentos em `docs/`, notebooks e materiais originais de consulta continuam disponíveis para versionamento. O notebook autocontido preserva a exposição e os resultados incorporados mesmo sem `outputs/`.

## Usar um modelo treinado

Grave suas formas em um arquivo NumPy com shape `(N,35)` ou `(35,)`. A ordem dos coeficientes é `g2metric.geometry.TRIPLES`: triplas lexicográficas de índices 0 a 6, equivalentes a `123,124,...,567` na notação matemática. Não inclua fator `3!` e não normalize a norma de cada forma.

```powershell
.\.venv\Scripts\python.exe -m g2metric predict --checkpoint outputs/full/checkpoints/mlp_augmented_n21000_seed11.pt --input minhas-formas.npy --output metricas.npy
```

A inferência aplica a padronização salva e verifica se as formas pertencem ao cone positivo da orientação fixada. O arquivo de saída contém as matrizes 7×7. O comando recusa sobrescrever um arquivo existente.

Para uma entrada de demonstração já disponível, substitua `minhas-formas.npy` por `outputs/example/phi0.npy`. A métrica exata dessa forma é a identidade.

Exemplo pela API, incluindo a referência exata:

```python
import numpy as np
from g2metric.geometry import phi0, metric_exact
from g2metric.training import load_predictor

phi = phi0()
assert np.allclose(metric_exact(phi), np.eye(7))
predict = load_predictor("outputs/full/checkpoints/mlp_augmented_n21000_seed11.pt")
g_estimado = predict(phi)
```

## Protocolo e interpretação

Os dados são `(A*φ₀, AᵀA)`, com `A = Q₁ diag(exp(s)) Q₂`, rotações Haar independentes e `sᵢ ∈ [−0,35; 0,35]`. A divisão é 70% treino, 15% validação e 15% teste **antes** da augmentação. O teste de extrapolação usa `sᵢ ∈ [−0,7; 0,7]` condicionado a pelo menos um valor fora do intervalo de treino.

A MLP 35–128–128–28 usa ReLU, Adam, early stopping e saída `L Lᵀ`, com diagonal de L positiva. A perda é a média dos erros relativos de Frobenius ao quadrado. A rede aumentada transforma entradas **e métricas** com rotações novas em cada minibatch. O teste nunca escolhe hiperparâmetros ou checkpoints.

Precisão, equivariância e escala são medidas separadamente: uma saída quase constante pode parecer equivariante e ainda ser uma aproximação ruim. A arquitetura garante simetria/positividade em aritmética exata, mas não equivariância. A distribuição original já é rotacionalmente invariante; a melhora com augmentação é uma hipótese experimental, não uma promessa.

## Documentação e estrutura

- [Notebook explicativo do projeto](notebooks/relatorio-projeto-g2.ipynb): texto, fórmulas, parâmetros, resultados e cinco figuras incorporadas; pode ser lido sem executar as células ou acessar outros arquivos.
- [Roteiro de estudo em oito sessões](docs/roteiro-de-estudo.md): ordem de leitura, função de cada pasta/arquivo, atividades e perguntas com respostas.
- [Planejamento, decisões e rastreabilidade dos requisitos](docs/analise-planejamento.md)
- [Registro de assistência de IA](docs/uso-ia.md)
- `g2metric/geometry.py`, `data.py`: álgebra exterior e dados sintéticos;
- `g2metric/models.py`, `training.py`: modelos, otimização e checkpoints;
- `g2metric/evaluation.py`, `reporting.py`, `experiment.py`: avaliações e execução reproduzível;
- `configs/`: configurações rápida e completa; `tests/`: verificações independentes.

O projeto não impõe condições de torção nula nem resolve equações diferenciais. Não há alegação de que a rede substitui ou acelera a fórmula exata.

## Ler e executar o notebook

Comece por `notebooks/relatorio-projeto-g2.ipynb` para a visão científica e use o roteiro para explorar o código aos poucos. As saídas dos oito exemplos já estão salvas, e as imagens estão anexadas às células. Os exemplos reexecutáveis são pequenos e não iniciam treinamento; os resultados medidos são incorporados ao documento.

No VS Code, abra o arquivo com suporte a notebooks Jupyter e, se quiser executar, selecione o Python de `.venv` como kernel. Num novo ambiente, instale o grupo opcional:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
```

`scripts/build_notebook.py` permite reconstruir e validar o documento a partir da execução registrada em `outputs/full/`. Esse script requer o repositório e os resultados; o notebook produzido pode ser lido e seus exemplos podem ser executados separadamente.
