# Roteiro para compreender o projeto G₂

O objetivo deste roteiro é conseguir explicar **qual pergunta o projeto investiga, como os dados atravessam o programa e o que os resultados permitem concluir**, sem precisar acompanhar cada operação de Python. O caminho sugerido tem oito sessões de 30–60 minutos; os comandos são opcionais.

Comece pelo [notebook do projeto](relatorio-projeto-g2.ipynb). Ele reúne a exposição matemática, o protocolo, os parâmetros, os resultados e os gráficos. Texto, figuras, resultados e exemplos didáticos estão incorporados ao arquivo: sua leitura não exige os dados nem os modelos salvos nas demais pastas. O presente roteiro é o mapa para, depois, relacionar essa exposição aos arquivos reais do projeto.

## A ideia em uma página

Queremos aproximar uma função conhecida: uma 3-forma positiva φ em um espaço vetorial orientado de dimensão 7 determina uma métrica gφ. O computador representa a entrada pelos **35 coeficientes independentes** da forma, pois há binomial(7,3) = 35 triplas. A resposta é uma matriz simétrica 7 × 7, com **28 entradas independentes**.

Há duas respostas de naturezas diferentes:

- **Resposta exata:** vem da fórmula de álgebra exterior ou, durante a geração dos dados, da identidade g(A*φ₀) = AᵀA. É a referência matemática.
- **Resposta aprendida:** uma regressão ridge ou uma rede recebe apenas os 35 coeficientes e tenta produzir a matriz correta. Seus parâmetros são ajustados com exemplos.

Comparam-se ridge, MLP e MLP com rotações durante o treinamento. A pergunta principal é se oferecer exemplos transformados melhora o respeito à geometria sem prejudicar a precisão. A mesma arquitetura de rede é usada nas duas MLPs. Não se está descobrindo uma fórmula desconhecida, resolvendo uma EDP ou construindo uma estrutura global sem torção.

```text
            Planejamento
                  ↓ orienta
             configs/full.json
                  ↓ controla
geometry.py → data.py → models.py + training.py → evaluation.py
                  ↑          ↑                         ↓
                  └──── experiment.py ─────────────────┤
                        coordena o fluxo               ↓
                                                 reporting.py
                                                      ↓
                                                 outputs/full/

__main__.py recebe o comando no terminal e chama a etapa escolhida.
tests/ verifica propriedades matemáticas e comportamento do programa.
docs/ reúne o planejamento, a apresentação, o relatório, o notebook e a documentação de apoio.
```

Uma **função** é uma operação que recebe entradas e devolve resultados. Um **módulo** é um arquivo `.py` que reúne operações relacionadas. Um **array** é uma tabela de números; `shape (N,35)` significa N exemplos, cada um com 35 coordenadas. Em PyTorch, um **tensor** também é uma tabela numérica, acompanhada das operações necessárias para calcular derivadas e treinar redes. Um **checkpoint** guarda um modelo já ajustado e informações necessárias para reutilizá-lo.

## Sessão 1 — Explicar a pergunta antes de abrir o código

**Objetivo:** descrever o problema em dois minutos.

Leia, nesta ordem, o [README principal](../README.md), a introdução do [notebook](relatorio-projeto-g2.ipynb) e o [planejamento comentado](analise-planejamento.md). Consulte o [PDF original](Planajamento%20de%20projeto.pdf) para separar o que ele pede das escolhas necessárias para implementar o experimento. O nome real do PDF tem a grafia `Planajamento`.

Identifique a entrada φ, a saída g, a referência exata e os três métodos comparados. Aprendizagem supervisionada significa ajustar uma função a pares de entrada e resposta conhecida; não há um agente descobrindo recompensas ou conversando com um modelo de linguagem durante o treino.

**Atividade:** escreva uma frase para cada item: “a rede recebe…”, “a rede deve devolver…”, “sabemos a resposta correta porque…”, “as rotações são introduzidas para…”. Evite usar apenas nomes de bibliotecas.

**Pode avançar quando:** conseguir distinguir o interesse experimental em aprendizado geométrico da capacidade de calcular g exatamente.

## Sessão 2 — Reconhecer as escolhas do experimento

**Objetivo:** compreender por que foram escolhidos esses métodos.

Releia as decisões descritas no [planejamento comentado](analise-planejamento.md) e abra [models.py](../g2metric/models.py). Nesta leitura, procure apenas os nomes `Standardizer`, `RidgeRegressor`, `MetricMLP` e `relative_frobenius_loss` e os comentários que os descrevem.

Pense na ridge como uma aplicação afim regularizada das coordenadas da entrada para as entradas independentes da matriz. Pense na MLP como uma família não linear de funções parametrizadas. PyTorch calcula as derivadas da perda em relação aos parâmetros; Adam usa essas derivadas para atualizá-los.

**Atividade:** explique por que um mapa algébrico com respostas conhecidas pode ser tratado por aprendizagem supervisionada e por que a ridge é uma referência útil para avaliar a MLP.

**Pode avançar quando:** souber explicar o papel de NumPy, PyTorch e Matplotlib e por que ridge é um comparador útil.

## Sessão 3 — A matemática que produz dados e respostas

**Objetivo:** entender de onde vem cada par (φ,g).

Leia as seções matemáticas do [notebook](relatorio-projeto-g2.ipynb), depois [geometry.py](../g2metric/geometry.py). Observe, nesta ordem, `phi0`, `pullback`, `sample_gl7` e `metric_exact`. Na primeira passagem, basta ler os comentários e reconhecer os argumentos e o resultado de cada função.

Parte-se de φ₀, cuja métrica é a identidade. Sorteia-se A = Q₁ diag(exp(s)) Q₂, com Q₁ e Q₂ rotações e s controlando os logaritmos dos valores singulares. O gerador produz φ = A*φ₀ e g = AᵀA. Assim, não precisa adivinhar quais vetores de 35 números representam formas positivas. A rede nunca recebe A ou s como atributos de entrada.

A função `metric_exact` reconstrói a métrica **a partir apenas de φ**, por uma computação de álgebra exterior. Sua comparação com AᵀA é uma verificação por caminhos distintos. As versões `pullback` e `pullback_torch` representam a mesma transformação, respectivamente em NumPy e PyTorch.

**Atividade:** para A = I, determine φ e g. Para A = 2I, determine os fatores que multiplicam φ₀ e a métrica. Relacione o segundo caso com g(tφ) = t^(2/3)g(φ).

**Pode avançar quando:** conseguir explicar por que a fórmula exata continua necessária em um projeto que treina redes.

## Sessão 4 — Separar geração, treino, validação e teste

**Objetivo:** compreender o desenho experimental e evitar conclusões contaminadas.

Abra [data.py](../g2metric/data.py), [configs/full.json](../configs/full.json) e o [manifesto dos dados](../outputs/full/dataset.json). Procure `generate_dataset` e identifique os conjuntos no dicionário devolvido.

Os 30.000 pares da distribuição usual são divididos em 21.000 para treino, 4.500 para validação e 4.500 para teste. Há mais 4.500 pares independentes para extrapolação. “ID” identifica o teste na distribuição de referência; “OOD” identifica o teste fora dela. No primeiro caso, cada sᵢ fica entre −0,35 e 0,35. No segundo, fica entre −0,7 e 0,7, exigindo que pelo menos um valor esteja fora da faixa inicial.

Treino ajusta parâmetros; validação escolhe alpha da ridge e o melhor estado da rede; teste mede o resultado depois dessas escolhas. A divisão acontece antes das rotações de treinamento. A padronização de cada coeficiente usa somente a média e o desvio padrão do subconjunto de treino escolhido. Ela preserva a informação de escala, diferentemente de dividir cada forma por sua própria norma.

**Atividade:** explique por que seria inadequado colocar uma forma no treino e uma cópia rotacionada dessa mesma amostra fonte no teste. Explique também por que a média da validação não participa da padronização.

**Pode avançar quando:** conseguir atribuir uma função distinta a cada um dos quatro conjuntos e distinguir `data.seed` de `seeds`.

## Sessão 5 — Entender os três modelos e o treinamento

**Objetivo:** saber o que muda durante um ajuste e o que distingue os métodos.

Releia [models.py](../g2metric/models.py) e abra [training.py](../g2metric/training.py). A sequência de interesse é `train_ridge`, `train_mlp`, `predictor`, `save_model`, `load_predictor`.

Na MLP, as larguras 35–128–128–28 indicam as coordenadas de entrada, duas camadas ocultas e a saída intermediária. Os 28 números formam uma matriz triangular L; a diagonal passa por uma função positiva, e a resposta é L Lᵀ. Isso impõe simetria e positividade em aritmética exata. A ridge prevê diretamente as 28 entradas de uma matriz simétrica, sem impor positividade.

Uma **época** percorre o conjunto de treino. Um **minibatch** é o grupo de exemplos usado em uma atualização. A perda é a média do quadrado do erro relativo de Frobenius. O treinamento pode parar quando a validação deixa de melhorar; o estado de menor perda de validação é restaurado ao final.

Na MLP aumentada, para cada exemplo do minibatch sorteia-se uma rotação C. Usa-se o novo par (C*φ, CᵀgC), antes de padronizar a entrada. Transformar apenas φ atribuiria, em geral, a resposta errada ao exemplo. As rotações são novas durante o treino e não criam uma divisão diferente dos dados fonte.

**Atividade:** desenhe um passo de treinamento com quatro caixas: predição → perda → derivadas → atualização. Marque o ponto em que entram as rotações e explique por que L Lᵀ não implica equivariância.

**Pode avançar quando:** conseguir distinguir treinar uma rede de usar uma rede já treinada para prever uma nova métrica.

## Sessão 6 — Ler resultados sem confundir precisão com simetria

**Objetivo:** interpretar cada diagnóstico por sua própria pergunta.

Leia [evaluation.py](../g2metric/evaluation.py), [resultados.md](resultados.md) e o [relatório completo](../outputs/full/report.md). Compare os quatro gráficos na pasta [figures](../outputs/full/figures).

| Diagnóstico | Pergunta que responde |
| --- | --- |
| Erro relativo de Frobenius | A matriz prevista está próxima da matriz verdadeira? |
| Mediana e percentil 95 | Qual é o erro central e quão grandes são os erros perto da cauda? |
| Defeito SO(7) | Prever após uma rotação concorda com rotacionar a previsão anterior? |
| Defeito GL⁺(7) | Essa compatibilidade também ocorre para mudanças de base não ortogonais? |
| Defeito de escala | F(tφ) concorda com t^(2/3)F(φ)? |
| Erro na escala transformada | F(tφ) se aproxima da resposta exata t^(2/3)gφ? |
| Falhas de positividade | Alguma previsão tem autovalor não positivo? |

No maior treino, as médias das medianas nas três sementes são:

| Modelo | Erro ID | Erro OOD |
| --- | ---: | ---: |
| Ridge | 36,58% | 62,29% |
| MLP | 25,81% | 43,57% |
| MLP com rotações | 24,99% | 42,79% |

Esses valores foram conferidos em [metrics.json](../outputs/full/metrics.json). São médias de três medianas, e não a mediana de um conjunto que reúne todos os resultados. A augmentação melhorou o erro e o defeito de rotação nas três sementes, mas os erros absolutos ainda são elevados. O experimento não demonstra equivariância exata, alta precisão ou ganho de velocidade sobre a fórmula conhecida.

**Atividade:** considere a função constante F(φ) = I. Mostre que ela respeita rotações e, ao mesmo tempo, pode prever uma métrica errada. Depois verifique sua lei de escala para t diferente de 1. Esse exemplo explica por que os diagnósticos são separados.

**Pode avançar quando:** conseguir justificar por que o pequeno defeito de rotação do ridge não o torna o melhor regressor.

## Sessão 7 — Seguir uma execução e localizar seus arquivos

**Objetivo:** saber o que acontece ao executar um comando e onde cada evidência é guardada.

Leia [__main__.py](../g2metric/__main__.py), depois `run_experiment` em [experiment.py](../g2metric/experiment.py) e a descrição de `build_report` em [reporting.py](../g2metric/reporting.py). O primeiro interpreta o comando; o segundo coordena dados, modelos e avaliações; o terceiro converte resultados já medidos em texto e figuras.

No experimento completo, 3 métodos × 3 tamanhos de treino × 3 sementes produzem **27 modelos**. Cada modelo é avaliado em ID e OOD, totalizando **54 registros** em `metrics.json`. Para uma mesma semente, os tamanhos 2.100, 7.000 e 21.000 formam subconjuntos aninhados do mesmo conjunto de treino. As sementes alteram a escolha/ordem dos exemplos e a aleatoriedade do ajuste; não geram três novos conjuntos completos de dados.

Abra uma sequência com o mesmo identificador: `mlp_augmented_n21000_seed11` em `histories/`, `checkpoints/` e `predictions/`. Você deve conseguir ligar a trajetória de treino, o modelo final e suas avaliações. Não é necessário abrir o conteúdo binário para compreender a ligação.

**Atividade:** localize a configuração usada, a época de melhor validação e a mediana de erro ID de um modelo, usando somente JSONs. Compare `best_epoch` com `epochs_run`: são conceitos distintos.

**Pode avançar quando:** souber encontrar um número do relatório nos registros que o originaram e souber usar uma pasta de saída nova para outro experimento.

## Sessão 8 — Entender as verificações e contar a história completa

**Objetivo:** explicar por que se pode confiar no fluxo e quais limites permanecem.

Leia primeiro os nomes dos testes, nesta ordem: [test_geometry.py](../tests/test_geometry.py), [test_data.py](../tests/test_data.py), [test_models.py](../tests/test_models.py), [test_pipeline.py](../tests/test_pipeline.py). Depois, consulte [uso-ia.md](uso-ia.md).

Os testes procuram erros de implementação: sinais e convenções, divisão dos dados, transformações corretas dos rótulos, saída positiva, escolha por validação e persistência dos modelos. Eles usam exemplos pequenos e controlados. Passar nos testes não prova que a rede generaliza para toda forma positiva; essa é uma questão diferente, investigada pelos experimentos e limitada às distribuições avaliadas.

**Atividade final:** prepare uma explicação de cinco minutos com este encadeamento: pergunta → dados → três métodos → escolha sem consultar testes → resultados → limitação principal. Cite pelo menos um teste matemático independente, um cuidado contra vazamento e uma conclusão que os resultados não autorizam.

**Conclusão do roteiro:** você compreende o projeto quando consegue dizer qual arquivo consultar para mudar uma configuração, verificar uma identidade, treinar, prever e conferir um resultado, mesmo sem saber implementar cada função.

## Mapa das pastas

| Pasta | Papel | Prioridade de leitura |
| --- | --- | --- |
| [docs/apresentacao](apresentacao) | Slides em PDF, fonte LaTeX e figuras da apresentação. | Síntese do projeto. |
| [docs/relatorio](relatorio) | Relatório em inglês, fonte LaTeX e figuras. | Texto de submissão. |
| [scripts](../scripts) | Ferramentas auxiliares de documentação, como a montagem do notebook a partir dos resultados registrados. | Depois de compreender o relato; não participa do treinamento. |
| [configs](../configs) | Valores que controlam geração, treinamento e avaliação. | Antes de executar experimentos. |
| [g2metric](../g2metric) | Código do projeto; é o pacote Python usado por `python -m g2metric`. | Ler por função científica, na ordem das sessões. |
| [tests](../tests) | Verificações automatizadas com casos conhecidos e pequenos. | Após entender o fluxo. |
| [docs](.) | Planejamento e notebook na raiz; decisões, resultados, registro de assistência e roteiro. | Começar pelo notebook e acompanhar o roteiro. |
| [outputs/full](../outputs/full) | Execução científica completa, com 27 ajustes. | Fonte dos números apresentados. |
| [outputs/smoke](../outputs/smoke) | Execução reduzida para verificar funcionamento; 600 pares ID, 90 OOD, uma semente e até 4 épocas. | Para conhecer arquivos sem confundir com resultados completos. |
| [outputs/example](../outputs/example) | `phi0.npy` e `metric.npy`, exemplos binários de entrada e saída de inferência. A resposta exata de φ₀ é I; uma previsão aprendida pode diferir. | Demonstração opcional. |
| `data/generated/` | Destino padrão do comando que gera apenas dados; pode ainda não existir. | Não é outra fonte de dados científicos externa. |
| `.venv/` | Ambiente Python local e bibliotecas instaladas. | Não estudar os milhares de arquivos das bibliotecas. |
| `.tools/` | Ferramentas locais de preparação do ambiente: Python, executáveis `uv`, caches e arquivo de instalação. | Infraestrutura; não contém o método matemático. |
| `.git/` | Banco interno do histórico Git. | Usar comandos Git, sem editar manualmente. |
| `.pytest_cache/` | Cache criado ao executar testes. | Não precisa estudar. |
| `__pycache__/` | Caches compilados do Python, dentro de pastas como `g2metric/` e `tests/`. | Não precisa estudar. |
| `tmp/` | Arquivos temporários de verificações de apresentação. `reporting-check/` e `reporting-check-fallback/` são ensaios do gerador de relatório. | Não usar como resultado científico principal. |

Arquivos como `.pyc`, pastas de cache e saídas repetidas seguem padrões; não representam novos métodos. Uma instalação pode acrescentar diretórios como `*.egg-info/`, que guardam metadados do pacote. Eles também não precisam entrar na primeira leitura.

## O que cada arquivo de código recebe e devolve

| Arquivo | Entrada | Trabalho principal | Saída |
| --- | --- | --- | --- |
| [geometry.py](../g2metric/geometry.py) | Coeficientes de formas, matrizes e parâmetros de amostragem. | Implementa φ₀, pullbacks, rotações, matrizes GL⁺ e fórmula exterior. | Formas transformadas, matrizes sorteadas e métricas exatas. |
| [data.py](../g2metric/data.py) | Quantidades de exemplos, limites de s e semente. | Gera pares, divide fontes, cria OOD e confere rótulos por fórmula independente. | Dicionário de arrays; grava/carrega `dataset.npz` e manifesto JSON. |
| [models.py](../g2metric/models.py) | Formas, parâmetros ajustáveis e alvos para a perda. | Define padronização, ridge, arquitetura MLP, saída L Lᵀ e perda. | Entradas padronizadas, matrizes previstas e valor de perda. |
| [training.py](../g2metric/training.py) | Treino, validação, padronizador, configuração e semente. | Ajusta modelos; seleciona pela validação; transforma exemplos; salva/carrega modelos. | Modelo ajustado, histórico e função de predição. |
| [evaluation.py](../g2metric/evaluation.py) | Função de predição, formas e métricas de teste, configuração. | Mede precisão, positividade, compatibilidade com transformações e escala. | Estatísticas agregadas e arrays de previsões/erros individuais. |
| [reporting.py](../g2metric/reporting.py) | Métricas, históricos, configuração e metadados. | Organiza tabelas, gráficos e comparação pareada entre redes. | `report.md` e quatro gráficos PNG; não retreina modelos. |
| [experiment.py](../g2metric/experiment.py) | Configuração e caminho de saída. | Valida configuração; coordena geração, ajustes e avaliações; registra ambiente e hashes. | Pasta completa de resultados. |
| [__main__.py](../g2metric/__main__.py) | Argumentos `run`, `generate` ou `predict` do terminal. | Encaminha para executar tudo, gerar só dados ou usar modelo salvo. | Artefatos da operação escolhida e mensagens no terminal. |
| [__init__.py](../g2metric/__init__.py) | Importação do pacote. | Identifica o pacote e declara sua versão. | Metadado `__version__`; não inicia treinamento. |

## Configurações, documentos e testes, arquivo por arquivo

| Arquivo | Para que serve |
| --- | --- |
| [README.md](../README.md) | Entrada operacional: instalação, execução, resultados e inferência. |
| [requirements.txt](../requirements.txt) | Dependências principais e de testes com versões mínimas; adequado para instalar o projeto em outros ambientes. |
| [requirements-lock.txt](../requirements-lock.txt) | Registro das versões exatas do ambiente usado no experimento, incluindo PyTorch CPU e ferramentas de PDF. É um retrato dessa execução, não uma lista a editar para escolher modelos. |
| [pyproject.toml](../pyproject.toml) | Nome/versão do pacote, versão mínima de Python, dependências, instalação, comando e configuração de testes. O grupo opcional `notebook` contém ferramentas para executar o `.ipynb`. |
| [.gitignore](../.gitignore) | Diz ao Git quais arquivos novos ignorar, como ambiente, caches, dados gerados, pesos e previsões. Ignorar não apaga arquivos locais; arquivos já rastreados têm tratamento próprio no Git. |
| [configs/full.json](../configs/full.json) | Protocolo completo com 30.000 exemplos ID e 27 ajustes. |
| [configs/smoke.json](../configs/smoke.json) | Protocolo curto para verificar o fluxo. Não substitui a execução completa. |
| [docs/relatorio-projeto-g2.ipynb](relatorio-projeto-g2.ipynb) | Relato auto-contido com conceitos, parâmetros, gráficos, resultados e pequenos exemplos. |
| [scripts/build_notebook.py](../scripts/build_notebook.py) | Lê resultados, configuração, metadados e figuras de `outputs/full/`; incorpora-os ao notebook, valida seu formato e executa os exemplos em uma pasta temporária isolada. Não treina nem importa `g2metric` nas células. |
| [docs/analise-planejamento.md](analise-planejamento.md) | Traduz cada requisito do PDF em decisões concretas e distingue escolhas adicionais. |
| [docs/resultados.md](resultados.md) | Interpretação científica da execução completa e suas limitações. |
| [docs/uso-ia.md](uso-ia.md) | Descreve a assistência de IA e distingue testes computacionais de revisão científica humana. |
| [docs/roteiro-de-estudo.md](roteiro-de-estudo.md) | Este roteiro: ordem de aprendizagem e mapa dos arquivos. |
| [tests/test_geometry.py](../tests/test_geometry.py) | Confere forma padrão, sinais, pullback, composição, naturalidade, escala, amostragem e acordo NumPy/PyTorch. |
| [tests/test_data.py](../tests/test_data.py) | Confere separação das fontes, limites ID/OOD, reprodução e gravação/leitura dos dados. |
| [tests/test_models.py](../tests/test_models.py) | Confere padronização, ajuste da ridge, saída positiva da rede, gradientes, perda e aprendizagem em caso simples. |
| [tests/test_pipeline.py](../tests/test_pipeline.py) | Confere seleção por validação, melhor checkpoint, transformação dos rótulos, persistência e distinção entre precisão e equivariância. |

### Como ler `full.json`

| Campos | Interpretação e valores da execução completa |
| --- | --- |
| `name` | Nome descritivo `full`; o caminho de saída é escolhido separadamente no comando. |
| `data.n_samples`, `data.n_ood`, `data.seed` | 30.000 pares ID, 4.500 OOD e semente 2026 para gerar/dividir dados. |
| `data.log_bound`, `data.ood_log_bound` | Limites 0,35 e 0,7 dos logaritmos dos valores singulares. |
| `data.validation_samples` | 64 exemplos por domínio para verificações numéricas da fórmula; não é o tamanho do conjunto de validação de treinamento. |
| `seeds`, `train_sizes` | Sementes 11, 22, 33 e tamanhos 2.100, 7.000, 21.000. |
| `hidden_dim`, `diagonal_eps` | Largura 128 das duas camadas ocultas e incremento positivo 10⁻⁵ na diagonal de L. |
| `batch_size`, `learning_rate` | 256 exemplos por minibatch e passo 0,001 do Adam. |
| `max_epochs`, `patience`, `min_delta` | Até 100 épocas; espera de 15 épocas sem melhora relevante de 10⁻⁵. Guarda-se sempre o menor valor de validação observado. |
| `ridge_alphas` | Candidatos 10⁻⁴, 10⁻², 1, 100 e 10.000; a validação escolhe a regularização. |
| `evaluation_samples`, `evaluation_seed` | 512 exemplos por domínio para diagnósticos geométricos; semente 8405. A precisão usa todos os 4.500 exemplos de cada teste. |
| `gl_log_bound`, `scale_factors` | Limite 0,2 para mudanças de base diagnósticas; fatores de escala 0,25, 0,5, 2 e 4. |
| `accuracy_tolerance_relative` | 0,05: admite até 5% de aumento relativo na mediana do erro para o critério comparativo. Não significa exigir erro absoluto inferior a 5%. |
| `threads`, `device` | Duas threads e CPU; controlam a execução computacional. |

## Decodificar os arquivos de uma execução

O mesmo esquema aparece em `outputs/full/` e `outputs/smoke/`. Cada pasta de execução deve ser lida como um conjunto: configuração, dados, modelos e resultados pertencem àquela execução.

| Arquivo ou padrão | Conteúdo e maneira de ler |
| --- | --- |
| `config.json` | Cópia dos parâmetros usados; preserva a configuração mesmo se `configs/full.json` for alterado depois. |
| `dataset.npz` | Arquivo NumPy compactado com formas, métricas, valores s, IDs e índices dos conjuntos ID, além dos dados OOD. É binário. |
| `dataset.json` | Manifesto legível: quantidades, convenções, precisão numérica, checagem da fórmula e hash do arquivo de dados. |
| `metadata.json` | Versões, plataforma, estado Git, hashes dos módulos e dos dados, verificações e status da execução. Um hash identifica o conteúdo de um arquivo; não é uma métrica de qualidade do modelo. |
| `metrics.json` / `metrics.csv` | Os mesmos resultados tabulares em formatos diferentes; um registro por método, semente, tamanho de treino e domínio. Valores de erro estão como frações: 0,25 corresponde a 25%. |
| `histories/ridge_n21000_seed11.json` | Alphas candidatos, perdas de validação e tempo da ridge. Ridge não tem épocas de rede; `best_epoch` é 0. |
| `histories/mlp_n21000_seed11.json` | Perdas por época, melhor época, quantidade de épocas executadas e tempo da rede sem rotações. |
| `histories/mlp_augmented_n21000_seed11.json` | Mesmo formato, para rede com rotações; o nome identifica o método. |
| `checkpoints/<método>_n<tamanho>_seed<semente>.pt` | Pesos ou coeficientes, padronização e identificação do treino; suficiente para carregar o preditor. É binário. |
| `predictions/<método>_n<tamanho>_seed<semente>_id.npz` | Previsões, alvos e erros individuais ID; índices dos exemplos usados nos diagnósticos e seus defeitos. |
| `predictions/<método>_n<tamanho>_seed<semente>_ood.npz` | Mesmo formato para OOD. O sufixo muda o domínio de avaliação, não o modelo treinado. |
| `figures/training_curves.png` | Perdas de treino e validação por época; mostra o maior treino e a primeira semente, 11, não todas as execuções. |
| `figures/learning_curves.png` | Efeito da quantidade de exemplos na mediana e no p95 ID/OOD; médias e desvios padrão entre sementes. |
| `figures/equivariance.png` | Diagnósticos SO(7) e GL⁺(7) no maior treino, resumidos entre sementes. |
| `figures/scaling.png` | Defeito de escala e erro contra a resposta exata, separados por fator. |
| `report.md` | Relato automático com tabelas, figuras, comparação pareada e rastreabilidade. |

Por exemplo, `mlp_augmented_n21000_seed11_ood.npz` significa: predições OOD da MLP com rotações, treinada com 21.000 fontes e semente 11. O treino usa dados armazenados em dupla precisão (`float64`) como origem; para as redes, entradas e alvos são convertidos a `float32`. “Dados em dupla precisão” não significa que toda operação da MLP usa essa precisão.

O `.gitignore` exclui toda a pasta `outputs/` de novos registros Git: dados, checkpoints, previsões, métricas, históricos, relatórios automáticos e gráficos. Também exclui `data/generated/`, arquivos binários de dados/modelos e logs. Esses arquivos permanecem no computador, mas não acompanham um clone do repositório; os links deste roteiro para `outputs/` dependem da execução local. Códigos, configurações de entrada, documentos em `docs/`, notebooks e materiais de consulta continuam versionáveis. O notebook autocontido pode ser lido sem as saídas locais; a inferência com checkpoints exige que o arquivo escolhido exista ou seja regenerado.

## Comandos opcionais para explorar

Execute no PowerShell, a partir da raiz `AI-Project`. O Python local evita depender de qual executável o comando genérico `python` encontra. Os comandos abaixo têm objetivos diferentes; não é necessário executá-los todos.

```powershell
# 1. Ver quais operações o programa oferece; não treina nada.
.\.venv\Scripts\python.exe -m g2metric --help

# 2. Executar os testes pequenos do software.
.\.venv\Scripts\python.exe -m pytest -q

# 3. Ler a configuração original e os parâmetros realmente usados.
Get-Content -Encoding UTF8 configs/full.json
Get-Content -Encoding UTF8 outputs/full/config.json

# 4. Inspecionar as métricas como tabela, sem abrir arquivos binários.
Import-Csv outputs/full/metrics.csv | Select-Object model, seed, train_size, domain, error_median
```

Para abrir o notebook no VS Code, use um visualizador compatível com Jupyter. A instalação de pacotes Python não instala automaticamente uma extensão do editor. Para executar as células, selecione o interpretador `.venv` como kernel; os exemplos do notebook são pequenos e não iniciam o treinamento completo. Se preparar um novo ambiente, a dependência opcional pode ser instalada por:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[notebook]'
```

Para estudar ou reexecutar as células, use o notebook pronto. **Regenerar o documento** é outra operação: o comando abaixo precisa dos resultados e figuras de `outputs/full/` e reconstrói/substitui `docs/relatorio-projeto-g2.ipynb`. Ele serve para atualizar o relato a partir das evidências registradas, não para produzir novos resultados científicos. Caso tenha feito anotações pessoais no notebook, salve antes uma cópia com outro nome.

```powershell
# Opcional: reconstruir o notebook a partir dos resultados existentes.
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

Para experimentar inferência usando os arquivos já disponíveis, o bloco abaixo cria um caminho novo a cada execução. Ele carrega um modelo existente; não ajusta seus parâmetros:

```powershell
$pastaExemplo = Join-Path 'outputs' ('estudo-predicao-' + [guid]::NewGuid().ToString('N'))
$arquivoSaida = Join-Path $pastaExemplo 'metricas.npy'
.\.venv\Scripts\python.exe -m g2metric predict --checkpoint outputs/full/checkpoints/mlp_augmented_n21000_seed11.pt --input outputs/example/phi0.npy --output $arquivoSaida
```

Quando quiser observar o fluxo inteiro, use **apenas o perfil curto** na primeira tentativa, também em pasta nova. Esse comando gera dados, treina modelos pequenos e produz outro relatório; os números dessa execução não devem substituir os resultados científicos completos:

```powershell
$pastaExperimento = Join-Path 'outputs' ('estudo-smoke-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m g2metric run --config configs/smoke.json --output $pastaExperimento
```

Para um experimento próprio, copie a configuração para um novo arquivo, mude uma escolha por vez e registre a motivação antes de olhar o teste. O programa recusa usar uma pasta de saída não vazia. O comando completo `run --config configs/full.json` é muito mais caro e não é necessário para compreender o projeto.

## O que pode ficar para uma segunda leitura

Deixe para depois os índices combinatórios da álgebra exterior, as expressões `einsum`, o algoritmo QR de amostragem, os detalhes de diferenciação automática, as opções de desenho dos gráficos e as rotinas internas de serialização. Primeiro entenda a operação matemática que cada trecho representa e como ela é verificada.

Você também pode deixar CNNs, transformers, agrupamentos, redução de dimensão, RL, difusão, PINNs e operadores para uma revisão mais ampla do curso. Eles não são componentes escondidos do programa. Na primeira passagem, leia os JSONs e os relatórios, sem tentar abrir `.npz`, `.npy` ou `.pt` como texto.

## Perguntas de compreensão, com respostas breves

1. **Por que 35 números de entrada e 28 de saída intermediária?** Há 35 coeficientes independentes de uma 3-forma em dimensão 7 e 28 posições no triângulo de uma matriz 7 × 7. Na MLP, essas 28 posições parametrizam L; na ridge, parametrizam diretamente g.
2. **Quem fornece a verdade para o treinamento?** A construção algébrica dá o par (A*φ₀, AᵀA); a fórmula exterior em φ confere esses rótulos independentemente.
3. **O que acontece para A = 2I?** O pullback multiplica a 3-forma por 2³ = 8, e a métrica por 2² = 4. Logo, g(8φ₀) = 8^(2/3)I = 4I.
4. **Por que separar validação de teste?** A validação participa da seleção. O teste deve avaliar a escolha pronta; usá-lo para selecionar faria sua avaliação ficar otimista.
5. **Por que a divisão precede a augmentação?** Descendentes da mesma fonte não devem aparecer como exemplos supostamente independentes em treino e teste.
6. **A padronização elimina a escala de φ?** Não. É uma transformação afim global invertível com estatísticas do treino; não iguala a norma de cada amostra.
7. **O que exatamente a augmentação transforma?** A forma e o rótulo: (φ,g) vira (C*φ,CᵀgC). O resultado continua sendo um par geometricamente correto.
8. **L Lᵀ garante equivariância?** Não. Com L invertível, garante simetria e positividade em aritmética exata. A compatibilidade com transformações é avaliada separadamente.
9. **Defeito geométrico pequeno garante boa aproximação?** Não. F(φ) = I respeita rotações, mas erra métricas não iguais a I; também falha na homogeneidade para t ≠ 1.
10. **O que OOD verifica aqui?** Generalização para outra faixa de logaritmos de valores singulares, com ao menos um fora da faixa ID. Não verifica todas as formas positivas possíveis.
11. **Por que 27 modelos e 54 registros de avaliação?** São 3 métodos × 3 tamanhos × 3 sementes; cada modelo tem dois registros, ID e OOD.
12. **Qual a diferença entre as sementes?** `data.seed` controla o conjunto e a divisão; `seeds` controla os subconjuntos/ajustes repetidos; `evaluation_seed` fixa exemplos e transformações dos diagnósticos.
13. **A perda do treino é o erro mediano da tabela?** Não. A perda é a média de erros relativos ao quadrado. A tabela resume medianas de erros relativos, sem o quadrado, nos testes.
14. **Uma tolerância relativa de 5% exige erro de até 5%?** Não. Um erro inicial de 20% poderia subir até 21% e ainda atender a essa tolerância comparativa. Ela não estabelece alta precisão.
15. **Os resultados demonstram vantagem definitiva da augmentação?** Mostram benefício nos pares de sementes e domínios avaliados, dentro desse orçamento. Não são prova universal nem teste de significância estatística.
16. **Qual é a diferença entre abrir o notebook e repetir o experimento?** O notebook já contém o relato e pequenas demonstrações. Repetir o experimento chama os módulos do projeto para gerar dados e ajustar novamente os 27 modelos.
17. **Qual arquivo mudar para outro tamanho de batch?** Uma nova cópia de `configs/full.json` ou de `configs/smoke.json`. Não é necessário alterar a definição da rede.
18. **Testes aprovados resolvem a questão matemática?** Eles dão evidência de correção da implementação em casos controlados. Não substituem demonstração, revisão humana nem avaliação de generalização.
