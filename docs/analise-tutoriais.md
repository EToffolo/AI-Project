# Análise dos tutoriais da MM845

Foram lidos integralmente os treze arquivos `material-consulta/tutoriais/README-01.md` a `README-13.md`. São descrições dos tutoriais, seus experimentos, resultados e exercícios; os notebooks citados nesses arquivos não estão incluídos nesta análise. Os números relatados nos READMEs são resultados dos autores do material, não resultados reproduzidos neste projeto.

Esta análise interpreta os tutoriais como extensão dos slides. Ela identifica ferramentas e práticas compatíveis com o projeto de aprender a métrica induzida por uma estrutura G2; não transforma todos os exemplos da disciplina em requisitos de implementação. Os requisitos específicos do projeto vêm do PDF de planejamento.

## Mapa do conteúdo

| Tutorial e fonte local | Conteúdo e ferramentas | Aplicação ao projeto |
|---|---|---|
| [01](../material-consulta/tutoriais/README-01.md) | Ambiente Python isolado, Miniforge/conda, NumPy, SciPy, Matplotlib, pandas, scikit-learn, SymPy, NetworkX, PyTorch, Jupyter, Git e assistentes de código; SageMath opcional | Ambiente reproduzível, geradores de dados, código modular, seeds explícitas e verificações matemáticas |
| [02](../material-consulta/tutoriais/README-02.md) | Amostragem de esferas, distâncias, ações de grupos, quaternions, concentração de medida, PCA local, mínimos quadrados e generalização | Definir a distribuição dos dados, controlar condicionamento e impedir vazamento de amostras aumentadas entre partições |
| [03](../material-consulta/tutoriais/README-03.md) | Mínimos quadrados, bases espectrais, Laplaciano de grafos, ridge/lasso, regressão logística, atributos invariantes e loop PyTorch | Baseline simples, normalização, diagnóstico de identificabilidade e interpretação cuidadosa de perdas |
| [04](../material-consulta/tutoriais/README-04.md) | MLPs, capacidade, regularidade, otimização não convexa, várias seeds e viés espectral | Rede supervisionada principal e comparação justa de modelos; resultados em várias inicializações |
| [05](../material-consulta/tutoriais/README-05.md) | CNNs, convoluções equivariantes a translação, pooling, inspeção de filtros e testes fora da distribuição | Princípio de impor somente simetrias verdadeiras; não há motivo para rasterizar os coeficientes de uma 3-forma |
| [06](../material-consulta/tutoriais/README-06.md) | Atenção, transformers de conjuntos, equivariância a permutações, batches por comprimento e ablações | Orienta escolha de arquitetura; os 35 coeficientes indexados de uma 3-forma não são um conjunto sem ordem |
| [07](../material-consulta/tutoriais/README-07.md) | k-means, clustering espectral, DBSCAN e componentes de curvas algébricas | Métodos exploratórios possíveis, sem necessidade para regressão com rótulo exato conhecido |
| [08](../material-consulta/tutoriais/README-08.md) | PCA, kernel PCA, t-SNE, distorção de vizinhanças/distâncias; UMAP opcional | Diagnóstico visual opcional, sem usar um gráfico bidimensional como evidência de exatidão |
| [09](../material-consulta/tutoriais/README-09.md) | MDPs, Bellman, Q-learning, Dijkstra e flips de triangulações | Reforça comparação com solução exata; RL não é necessário para este mapeamento supervisionado |
| [10](../material-consulta/tutoriais/README-10.md) | Augmentation, atributos invariantes, camadas equivariantes, ações SO(3), GNNs e limites de expressividade | Distinguir saída covariante de rótulo invariante; medir defeito de equivariância sob mudança de base |
| [11](../material-consulta/tutoriais/README-11.md) | Modelos de difusão, score, denoising, amostragem reversa e difusão intrínseca no toro | Separar geração de dados de erro de aprendizado; difusão não é necessária quando há gerador exato |
| [12](../material-consulta/tutoriais/README-12.md) | PINNs, autodiff, Adam/L-BFGS, restrições fortes/fracas, FEM, Laplace–Beltrami e Deep Ritz | Incorporar restrições geométricas na saída e validar fora da perda de treino; o problema pontual não requer uma PDE |
| [13](../material-consulta/tutoriais/README-13.md) | PINNs espaço-temporais, causalidade, janelas temporais, fluxo por curvatura, DeepONet e camada espectral FNO | Verificações por identidades, testes fora da família treinada e entrega científica reproduzível |

## Leitura detalhada

### Tutorial 01 — ambiente e reprodução

O ambiente proposto usa Python 3.12, bibliotecas científicas e PyTorch, suficientes para executar os exemplos em CPU. O documento ensina terminal, Git/GitHub, Jupyter e VS Code, e propõe `src/`, `data/`, `notebooks/`, `results/`, README e especificação de ambiente. O objetivo de reprodução é criar o ambiente e executar o experimento. Os dados gerados e pesos não devem ser confundidos com código-fonte. O uso de agentes é acompanhado de especificações precisas, pequenos passos, leitura do código e testes contra casos conhecidos. Não é necessário instalar todas as ferramentas opcionais.

### Tutorial 02 — amostragem, geometria e generalização

Esferas oferecem distribuição, dimensão, distâncias e simetrias conhecidas para verificar resultados. O tutorial compara amostragem uniforme e parametrizações enviesadas, distâncias cordais e geodésicas, transformações SO(3), S3/quaternions/Hopf e PCA local. Mostra a curva viés–variância e double descent em regressão polinomial. A distribuição geradora é parte do modelo científico, não apenas um detalhe de implementação. Uma divisão aleatória depois de gerar várias transformações da mesma amostra pode vazar informação. O projeto deve explicitar como produz formas positivas, quais escalas e condicionamentos cobre e quais amostras são realmente independentes.

### Tutorial 03 — modelos lineares e escolha de atributos

Mínimos quadrados aparecem como projeção; a base determina condicionamento, velocidade de otimização e legibilidade dos coeficientes. A base de autofunções do Laplaciano e a regularização de Dirichlet conectam álgebra linear e geometria. Na regressão de curvas, atributos adequados recuperam uma fórmula de área com coeficientes interpretáveis, enquanto coordenadas brutas falham. O tutorial também apresenta ridge, lasso, classificação e o loop de treinamento PyTorch. Perda pequena não implica coeficientes corretamente identificados. Para o projeto, um baseline linear deve ser medido junto da MLP, e a fórmula geométrica conhecida deve servir de referência independente.

### Tutorial 04 — redes como espaços de aproximação

Um MLP aprende a transformação de atributos e perde a convexidade da regressão linear. A comparação com polinômios a número de parâmetros semelhante mostra que redes não são automaticamente melhores: a regularidade e a existência de uma base adequada importam. Diferentes seeds produzem diferentes mínimos e erros. O viés espectral faz componentes suaves serem aprendidas antes de detalhes oscilatórios. No projeto, a avaliação deve relatar mais de uma inicialização quando fizer afirmações comparativas, guardar curvas de treino/validação e considerar o mapa analítico conhecido antes de atribuir vantagem à rede.

### Tutorial 05 — convolução e simetria

O tutorial verifica diretamente a comutação entre convolução e translação, e o papel do pooling na invariância. A comparação CNN/MLP em curvas rasterizadas distingue desempenho na distribuição de treino de robustez a translações. Mostra também que filtros podem detectar ocupação, não apenas bordas. A lição transferível é testar a simetria alegada. Os coeficientes de uma 3-forma em uma base não possuem a simetria de translação de uma imagem, de modo que CNN não é uma escolha justificada apenas porque aparece no curso.

### Tutorial 06 — atenção em conjuntos

Atenção é descrita como média dependente dos dados, equivarante a permutações, seguida de pooling para tarefas invariantes. O exemplo de diâmetro de nuvens de pontos admite número variável de elementos, enquanto um MLP achatado depende de largura e ordem fixas. O tutorial agrupa comprimentos iguais para não preencher artificialmente os conjuntos. Mapas de atenção devem ser confrontados com ablações antes de receber interpretação causal. No projeto G2, trocar arbitrariamente a ordem dos 35 coeficientes altera a 3-forma; a indexação faz parte dos dados. Um transformer de conjunto não deve apagar essa informação.

### Tutorial 07 — agrupamentos e hipóteses geométricas

k-means, clustering espectral e DBSCAN procuram objetos diferentes: grupos próximos de centros, regiões fracamente conectadas e componentes densas. Os exemplos de círculos concêntricos e curvas elípticas tornam esses pressupostos testáveis. O número de componentes pode ser ambíguo perto da degeneração ou em estruturas finas. Tais métodos podem investigar regimes de dados, mas não são requisito nem substituto da regressão supervisionada quando a métrica exata já fornece o rótulo.

### Tutorial 08 — redução de dimensão e distorções

PCA estima a dimensão do espaço linear ocupado, não necessariamente a dimensão intrínseca. Kernel PCA depende da escolha de kernel e nem sempre põe a informação desejada nas primeiras componentes. O toro ilustra obstruções geométricas à visualização sem distorção. t-SNE privilegia vizinhanças e não preserva automaticamente distâncias globais, tamanhos ou números de componentes. Um gráfico desses pode ilustrar a distribuição das formas, mas não comprova boa cobertura do domínio, generalização ou recuperação exata da métrica.

### Tutorial 09 — aprendizado por reforço

O material constrói MDPs em triangulações e usa Q-learning para caminhos mínimos e flips. Dijkstra e o teorema de Delaunay fornecem referências exatas. O valor aprendido só é confiável onde as trajetórias exploram; a escolha de desconto muda o objetivo. Mesmo quando RL reencontra o ótimo, pode ser mais lento que o algoritmo clássico e não fornece prova. O problema G2 proposto é um mapa direto entre tensores, sem sequência necessária de decisões; RL acrescentaria uma formulação desnecessária.

### Tutorial 10 — invariância, equivariância e informação

Compara aumento de dados, descritores invariantes e arquitetura equivarante em orçamento semelhante, incluindo curvas de eficiência amostral. A simetria pode ser respeitada e ainda assim faltar uma transformação de atributos útil. Exemplos de grafos mostram limites distintos de espectros e message passing: nenhum é universalmente suficiente para todo rótulo. Para G2, a matriz da métrica muda com a base; é uma saída covariante, não um escalar invariante. Apenas fornecer autovalores ou outros invariantes da entrada não determina em geral as componentes orientadas da saída. É necessário declarar a ação do grupo, as convenções de índices e a lei de transformação usada nos testes.

### Tutorial 11 — modelos de difusão

A cadeia de corrupção, o score, o treinamento por denoising e o amostrador reverso são analisados separadamente em um círculo com resposta conhecida. A perda tem piso irredutível e erro pequeno em ruídos altos não garante bom score em ruídos baixos. O toro mostra a vantagem de construir a restrição de variedade na própria dinâmica. Para G2, existe um gerador algébrico de formas positivas a partir de mudanças de base; um modelo generativo treinado não é necessário para criar os dados supervisionados. A separação entre erro de geração, erro de aprendizado e erro de avaliação é, entretanto, essencial.

### Tutorial 12 — PINNs e verificação independente

Problemas de Poisson no disco e na esfera com soluções fabricadas permitem medir erro verdadeiro. O documento ensina autodiff, condições de contorno incorporadas ao ansatz, Adam seguido de L-BFGS, gauge fixing, FEM e Deep Ritz. Mostra que resíduo baixo pode acompanhar solução errada e que a quadratura altera a energia minimizada. Para o projeto, impor simetria e positividade da métrica por parametrização é uma aplicação do princípio de restrições fortes. Isso não torna a rede uma PINN: o mapa algébrico pontual não possui uma PDE a resolver. A fórmula exata e identidades tensoriais devem validar a saída.

### Tutorial 13 — dinâmica, operadores e entrega

Acrescenta tempo a PINNs, analisa falha de propagação, pesos causais, atributos de Fourier e janelas temporais. O fluxo por curvatura é verificado por leis de área e raios, inclusive quando não há solução explícita. DeepONet e uma camada FNO aprendem operadores; uma base que contém o operador verdadeiro vence uma arquitetura genérica. Resolução de malha maior não recupera modos nunca excitados no treino. O paralelo em G2 é testar escalas, anisotropias e condicionamentos novos, sem afirmar validade em toda a órbita a partir de um domínio estreito.

O fechamento do tutorial pede uma pergunta clara, dados compreendidos, método apropriado, baseline simples e verificação independente. Registra apresentação de dez minutos, relatório de cinco páginas e repositório reproduzível; esses são requisitos de entrega descritos na cópia local do curso, não uma confirmação de cronograma externo atualizado.

## Decisões recomendadas para o projeto G2

1. **Formulação:** aprender o mapa pontual dos 35 coeficientes independentes de uma 3-forma positiva para uma matriz métrica simétrica 7 × 7; o planejamento determina a variante final. Declarar ordem dos coeficientes, convenção da forma padrão e ação de mudança de base.
2. **Dados:** produzir pares por transformações explicitamente amostradas; registrar seed, escala, anisotropia e número de condição. Não tratar a distribuição escolhida como uniforme em um espaço não compacto.
3. **Referências:** comparar com a fórmula geométrica exata, predição constante e regressão linear/ridge, além da rede PyTorch. Um método exato pode ser mais rápido e preciso; isso é um resultado válido.
4. **Arquitetura:** MLP é compatível com os tutoriais 03–04. Uma saída construída por fator triangular e produto com sua transposta pode garantir positividade e simetria. Essa parametrização deve ser descrita como aplicação do princípio de restrições fortes, não como uma técnica explicitamente ensinada nos READMEs.
5. **Validação:** dividir treino/validação/teste antes de qualquer aumento derivado da mesma amostra; escolher hiperparâmetros na validação. Medir erro relativo matricial, autovalores, positividade, casos conhecidos e defeito de equivariância em transformações independentes.
6. **Generalização:** separar teste na distribuição, teste sob transformação e teste fora do intervalo de escala/condicionamento. Medir várias seeds para conclusões comparativas; um treinamento curto apenas verifica a execução do pipeline.
7. **Reprodução:** fornecer configuração, gerador, comando de treino/avaliação, versões e resultados identificados por configuração e seed. Não apresentar números dos tutoriais como resultados do projeto.

## Limites e ressalvas matemáticas

- Todas as 3-formas positivas em um espaço orientado de dimensão sete pertencem à mesma órbita de GL+(7). Portanto, não faz sentido prometer separar treino e teste por órbitas desse grupo. O controle de vazamento deve agrupar descendentes da mesma amostra base; outras noções de órbita só devem ser usadas depois de especificar o subgrupo.
- A frase final do tutorial 10 sobre completude do espectro da covariância para nuvens de pontos precisa de restrição ao contexto do exemplo. O espectro da covariância não determina uma nuvem arbitrária a menos de O(3). Não se deve transportar essa afirmação para formas G2.
- Autodiff diferencia o programa implementado até os limites da aritmética utilizada; não prova que o programa expressa o operador matemático correto. A verificação contra casos exatos continua necessária.
- Restrições positivas na saída não garantem equivariância. Aumentar dados não garante uma identidade exata fora das amostras observadas.
- Os READMEs citam exercícios e leituras adicionais; essa análise não presume que notebooks ausentes ou artigos citados tenham sido executados ou examinados.
