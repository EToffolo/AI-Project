# Análise dos materiais de aula MM845

Este documento registra a leitura dos 15 arquivos de `material-consulta/slides`: os 13 PDFs `lecture_01.pdf` a `lecture_13.pdf`, o pôster em PDF e sua versão em PNG. Os PDFs das aulas têm 12 páginas, exceto as aulas 3, 5 e 6, que têm 11 páginas. O pôster PDF tem uma página. O texto de todas as páginas foi examinado; o pôster nos dois formatos e as páginas centrais sobre regularização, equivariância e entregáveis também foram inspecionados visualmente.

O planejamento local foi usado para decidir a pertinência de cada técnica. Seu nome real no projeto é [`Planajamento de projeto.pdf`](../material-consulta/Planajamento%20de%20projeto.pdf). A análise dos tutoriais é complementar e deve ser lida em conjunto com esta.

## Conclusão para a implementação

O projeto proposto se encaixa diretamente nas aulas 1, 2, 3, 4 e 10: código científico reprodutível; geração de pares supervisionados e avaliação independente; regressão ridge como referência; MLP em PyTorch; aumento de dados por simetrias e teste de equivariância. As aulas 11 a 13 reforçam a escolha de representações adequadas, a imposição de restrições geométricas e a necessidade de verificação independente.

O conteúdo da disciplina oferece um conjunto de métodos; não impõe que todos sejam empregados, nem apresenta uma lista formal de bibliotecas proibidas. Aqui, uma técnica estar no programa significa que tem suporte no material. A pertinência ao projeto é uma segunda decisão: CNNs, transformers, clustering, modelos generativos, RL, PINNs e operadores neurais estão no programa, mas não são necessários para o problema algébrico pontual definido no planejamento. Introduzi-los no experimento principal mudaria o escopo sem responder melhor à comparação prevista.

## Registro por documento

### 1. `lecture_01.pdf`: Coding Foundations: Python, Git, & AI Agents

**Conteúdo:** computação como instrumento de experimentação matemática; Python; vetorização; ecossistema NumPy, SciPy, scikit-learn, Matplotlib, PyTorch e SageMath; ambientes isolados; sementes; organização de repositório; Git/GitHub; assistentes e agentes de código. A aula distingue geração de exemplos, teste de conjecturas e demonstração matemática.

**Aplicação:** usar funções pequenas com convenções e formas de tensores explícitas; dados, código, configurações e resultados separados; registrar ambiente e aleatoriedade; salvar resultados para que gráficos possam ser refeitos sem treinar novamente. Empregar assistência de IA para programação é expressamente compatível com a disciplina.

**Limite:** código que executa sem erros não está automaticamente matematicamente correto. O material pede testes em casos conhecidos e computações independentes, além da leitura crítica do código produzido por agentes. Referências: páginas 4 a 6 e 9 a 11.

### 2. `lecture_02.pdf`: Mathematical Foundations of Machine Learning

**Conteúdo:** aprendizagem como aproximação, otimização e estatística; risco populacional e empírico; espaços de hipóteses e funções de perda; aprendizagem supervisionada, não supervisionada e por reforço; generalização; viés e variância; hipótese de variedade; dimensão e concentração; SGD, momentum, RMSProp e Adam; protocolo de dados.

**Aplicação:** formular `F: Λ³(R⁷)* → Sym⁺(7)` como regressão supervisionada sobre o domínio positivo; usar pares sintéticos com resposta conhecida; escolher perda relativa de Frobenius; separar treino, validação e teste; ajustar a padronização apenas no treino; registrar métricas e configurações.

**Limite:** seleção de hiperparâmetros e parada antecipada usam validação, nunca teste. A página 11 alerta para vazamento quando cópias simétricas caem em partições diferentes. Neste projeto, separar pares originais antes de qualquer augmentation e manter cada derivado junto de sua fonte. Não interpretar literalmente esse alerta como separar órbitas de `GL⁺(7)`: todo o domínio positivo do planejamento é uma única órbita desse grupo. Referências: páginas 4, 6, 10 e 11.

### 3. `lecture_03.pdf`: Linear Models, Optimisation & Regularisation

**Conteúdo:** mínimos quadrados como projeção ortogonal; regressão logística e softmax; gradiente como discretização de fluxo; convexidade, suavidade e condicionamento; minibatches; regularização ridge e lasso; mapas de atributos; laço de treinamento PyTorch.

**Aplicação:** implementar ridge para os 28 coeficientes independentes da matriz simétrica; incluir intercepto e padronização; selecionar a regularização pela validação. Este é o comparador simples explicitamente previsto no planejamento e recomendado pela aula antes de redes mais complexas.

**Limite:** resolver o problema linear por fatoração ou sistema linear estável, sem formar a inversa explicitamente. A regularização não torna uma saída simétrica automaticamente positiva definida. Não corrigir silenciosamente o ridge por projeção no cone positivo, pois isso alteraria a referência e esconderia as falhas que o planejamento manda contar. Aulas 3, páginas 3, 8 e 9.

### 4. `lecture_04.pdf`: Neural Networks: Theory, Architecture, Training

**Conteúdo:** MLPs como composições de mapas afins e ativações; ReLU, GELU, SiLU e tanh; aproximação universal; autodiferenciação e retropropagação; inicialização; normalização; Adam; curvas de treino/validação; regularização, dropout, parada antecipada e aumento de dados; redes para grandezas geométricas; boas práticas com `nn.Module`.

**Aplicação:** MLP `35 → 128 → 128 → 28` com ReLU, treinada por Adam; saída convertida em fator triangular `L`, com diagonal positiva via softplus, e métrica `L Lᵀ`; usar a mesma arquitetura nos braços com e sem augmentation; restaurar o melhor estado segundo a validação; registrar curvas e repetir por sementes.

**Limite:** aproximação universal não garante otimização bem-sucedida, eficiência nem extrapolação. O exemplo genérico de augmentation na página 9 preserva rótulos de tarefas invariantes; para uma métrica, o rótulo deve acompanhar a ação tensorial `g → Cᵀ g C`. Manter `g` fixo ao rotacionar uma forma seria incorreto. A parametrização triangular vem do planejamento específico, e sua adequação segue o princípio geral de respeitar a geometria; a aula não ensina um modelo G₂ pronto. Referências: páginas 3 a 7 e 9 a 11.

### 5. `lecture_05.pdf`: Convolutional Neural Networks

**Conteúdo:** sinais em grades; localidade; convolução e equivariância por translação; canais, stride, padding e campo receptivo; pooling local e global; LeNet, AlexNet, VGG, Inception, ResNet, U-Net e ConvNeXt; filtros como operadores discretos; dados geométricos em grades.

**Aplicação:** aproveitar a distinção entre equivariância e invariância e a regra de escolher arquitetura segundo a estrutura real dos dados.

**Limite:** os 35 coeficientes de uma 3-forma não constituem imagem com simetria de translação. Organizar arbitrariamente os coeficientes em uma grade e aplicar CNN introduziria um prior sem justificativa. A página 9 explicitamente desaconselha CNNs quando falta a estrutura de grade/translação. A regressão de um tensor também não deve terminar em pooling que apague sua transformação geométrica. Referências: páginas 4, 6 e 9.

### 6. `lecture_06.pdf`: Transformers & Attention

**Conteúdo:** dados relacionais de tamanho variável; atenção com queries, keys e values; múltiplas cabeças; equivariância por permutação; posições; blocos encoder/decoder; resíduos e LayerNorm; LLMs como ferramentas de matemática; tokens geométricos e pooling para saídas invariantes.

**Aplicação:** fundamenta o uso crítico de IA para programar e documentar. A advertência de conferir resultados, referências e derivações é pertinente ao desenvolvimento inteiro.

**Limite:** aqui o vetor de entrada tem dimensão fixa e base determinada, sem necessidade de modelar conjuntos de tamanho variável. Atenção equivarante por permutações não implica equivariância pela representação de `SO(7)` em 3-formas. Um transformer seria possível como extensão, mas não faz parte da comparação proposta. Referências: páginas 5, 8 a 10.

### 7. `lecture_07.pdf`: Unsupervised Learning - Clustering

**Conteúdo:** k-means e seu objetivo; inicializações, elbow e silhouette; kernel k-means; clustering hierárquico e linkage; DBSCAN/HDBSCAN; relação com componentes e homologia persistente; clustering espectral com Laplacianos; métricas extrínsecas, intrínsecas e invariantes; validação por estabilidade e verdade conhecida; complexidade computacional.

**Aplicação:** oferece maneiras opcionais de explorar regimes de dados e casos difíceis. A principal lição utilizada é escolher representação e distância coerentes com a geometria.

**Limite:** encontrar clusters não substitui o aprendizado supervisionado nem avalia a qualidade da métrica prevista. Agrupamentos podem refletir escala, representação ou algoritmo. Métodos espectrais e kernels densos têm custos quadráticos ou cúbicos e não são necessários para o conjunto de 30.000 exemplos. Referências: páginas 3, 9 e 11.

### 8. `lecture_08.pdf`: Dimensionality Reduction - PCA, Kernel PCA & Nonlinear Visualisation

**Conteúdo:** estrutura preservada por embeddings; PCA como problema variacional/espectral e SVD; variância explicada; whitening; kernel PCA; MDS e Isomap; t-SNE e UMAP; limitações de visualizações e cuidados de interpretação.

**Aplicação:** PCA poderia ser diagnóstico suplementar da distribuição de dados, acompanhado da variância explicada. Os gráficos exigidos no planejamento são principalmente erros, curvas de aprendizado e defeitos de simetria; não exigem redução de dimensionalidade.

**Limite:** não substituir os 35 coeficientes por duas dimensões no experimento principal. Distâncias e tamanhos aparentes de clusters em t-SNE/UMAP não são evidência de estrutura geométrica, boa generalização ou equivariância. A escala física da forma deve ser preservada mesmo quando se aplica uma transformação afim de padronização. Referências: páginas 3 a 5, 8, 9 e 11.

### 9. `lecture_09.pdf`: Introduction to Reinforcement Learning

**Conteúdo:** agentes e ambientes; MDPs; política, valor e equações de Bellman; métodos baseados em valor, política e modelo; Q-learning, DQN, replay e redes alvo; REINFORCE, actor-critic e PPO; busca de contraexemplos; desenho de recompensas; comparação com buscas exaustiva, aleatória, local e evolutiva; Gymnasium.

**Aplicação:** nenhuma etapa do experimento principal demanda ações sequenciais ou recompensa atrasada. A aula fornece o critério para descartar RL neste caso.

**Limite:** não transformar uma regressão com rótulos exatos disponíveis em problema de RL. Esse acréscimo teria custo, instabilidade e sensibilidade a sementes sem servir à hipótese proposta. Referências: páginas 3 e 11.

### 10. `lecture_10.pdf`: Geometry-Aware ML I - Equivariance & Symmetry

**Conteúdo:** definições de invariância e equivariância; ações nos espaços de entrada e saída; hipóteses equivariantes; intertwiners e não linearidades compatíveis; representações irredutíveis; CNNs de grupo, steerable e esféricas; GNNs, message passing, GCN, GAT, GIN e limite de Weisfeiler-Leman; DeepSets, nuvens de pontos e redes E(3)-equivariantes; três estratégias de incorporar simetria.

**Aplicação:** é a principal base conceitual da comparação: representar a ação `φ → C*φ`, transformar as saídas por `g → CᵀgC`, treinar com rotações e medir o defeito em transformações novas. Manter separados os testes de rotação, mudança de base não ortogonal e homogeneidade.

**Limite:** aumento de dados com uma MLP genérica só tenta aprender a equivariância; não a garante exatamente. Positividade por Cholesky tampouco garante equivariância. O planejamento exclui expressamente uma arquitetura exatamente `GL⁺(7)`-equivariante. As figuras da página 10 ilustram sobretudo saída invariante, e devem ser adaptadas à saída tensorial deste projeto, conforme a definição geral da página 3. Referências: páginas 3 a 5 e 10.

### 11. `lecture_11.pdf`: Geometry-Aware ML II - Manifolds & Diffusion Models

**Conteúdo:** representação de variedades por embeddings, atlas ou bases espectrais; potenciais de métricas Calabi-Yau; geração de distribuições; normalising flows; referência a VAEs e GANs; ruído progressivo, score, regressão de denoising, processo reverso e difusão em variedades.

**Aplicação:** reforça que a representação de funções, formas e métricas deve obedecer à transformação apropriada. O projeto usa dados sintéticos gerados por uma construção matemática conhecida e não precisa aprender sua distribuição com um modelo generativo.

**Limite:** aprender a métrica de uma 3-forma positiva em `R⁷` não é construir uma métrica Calabi-Yau nem resolver a existência global de métricas em variedades. A restrição de positividade precisa ser assegurada e verificada explicitamente; a discussão mais cuidadosa aparece na aula 13, página 5. Referências: páginas 3 a 6 e 11.

### 12. `lecture_12.pdf`: Physics-Informed Neural Networks for Geometry I

**Conteúdo:** métodos clássicos para EDPs; PINNs e pontos de colocação; resíduos e condições de contorno; autodiferenciação; suavidade; restrições suaves e impostas pelo ansatz; Poisson no disco; Laplace-Beltrami; Deep Ritz; problemas inversos; desequilíbrio de perdas, viés espectral e verificação independente.

**Aplicação:** o princípio de incorporar restrições no ansatz apoia a saída positiva definida. O princípio de verificar além da perda de treino apoia os testes de identidades algébricas, escala, autovalores e erro em dados independentes.

**Limite:** o planejamento trata uma aplicação algébrica pontual e explicitamente exclui EDPs e condições de ausência de torção. Não há motivo para adicionar PINNs. A ReLU é adequada à regressão planejada; a advertência da página 5 sobre suas segundas derivadas refere-se ao uso de resíduos diferenciais de segunda ordem. Referências: páginas 4 a 6 e 11.

### 13. `lecture_13.pdf`: PINNs for Geometry II & Neural Operators

**Conteúdo:** evolução espaço-temporal; fluxos geométricos; problemas não lineares e seleção do ramo geométrico correto; equilíbrio de perdas e causalidade; características de Fourier; decomposição XPINN/FBPINN; DeepONet; Fourier Neural Operator; operadores em variedades; síntese do curso e entregáveis.

**Aplicação:** a página 5 reforça que satisfazer um resíduo não basta para garantir convexidade ou positividade. As páginas 11 e 12 pedem pergunta clara, método correspondente aos dados, comparador simples, verificação independente, código reprodutível e explicação inclusive de resultados negativos.

**Limite:** não há família de soluções de EDP ou fluxo a aproximar neste projeto; DeepONet/FNO e PINNs espaço-temporais não são requeridos. A entrega acadêmica indicada no slide final é apresentação de 10 minutos, relatório de 5 páginas e repositório reprodutível. O slide registra apresentação em 01/10 e relatório em 04/10; são datas do material fornecido, não confirmação externa de calendário. Referências: páginas 5, 8 a 12.

### 14. `AI_course_poster.pdf`

O pôster resume os cinco blocos do curso: fundamentos computacionais e matemáticos; aprendizagem supervisionada; métodos não supervisionados e reforço; aprendizagem geométrica; EDPs geométricas e computação científica. Registra 13 aulas e 13 tutoriais, 30 horas, 2 créditos e projeto individual, com plano de 2 páginas, relatório de 5 páginas e apresentação de 10 minutos. Confirma o enquadramento de uma investigação experimental de IA em geometria; não prescreve usar todos os blocos.

### 15. `AI_course_poster.png`

A inspeção visual confirma que é a versão em imagem do mesmo pôster: mesmo programa, estrutura e entregáveis. É um segundo arquivo coberto pela análise, sem adicionar um método ou requisito distinto do PDF.

## Matriz de decisões para este projeto

| Decisão | Sustentação no curso | Papel no projeto |
| --- | --- | --- |
| Python, NumPy e PyTorch | Aulas 1, 3 e 4 | Geração matemática, regressão e treinamento |
| Matplotlib e resultados persistidos | Aulas 1 e 4 | Gráficos reproduzíveis de erros e curvas |
| Seeds, ambiente e configurações registrados | Aulas 1, 2 e 4 | Reprodutibilidade e variação entre execuções |
| Separação treino/validação/teste | Aula 2 | Seleção sem contaminação do teste |
| Ridge com regularização validada | Aula 3 | Referência linear prevista no plano |
| MLP com ReLU, Adam e early stopping | Aula 4 | Regressor não linear principal |
| Saída `L Lᵀ` com diagonal positiva | Planejamento; princípio de restrições nas aulas 12 e 13 | Positividade estrutural da previsão |
| Transformações de `SO(7)` durante treino | Aulas 4 e 10; ação concreta no planejamento | Comparação controlada da hipótese de augmentation |
| Testes separados de erro, equivariância e escala | Aulas 2, 10, 12 e 13; fórmulas do planejamento | Evitar confundir acurácia com respeito à geometria |
| CNNs, atenção, GNNs, clustering e PCA | Aulas 5 a 10 | Métodos ensinados, sem necessidade no núcleo proposto |
| RL, difusão, PINNs e operadores | Aulas 9, 11 a 13 | Métodos ensinados para outras tarefas; fora do núcleo proposto |

## Cuidados que orientam a leitura dos resultados

1. **Duas computações para a resposta geométrica.** O gerador usa `g = AᵀA`; a verificação deve obter a métrica diretamente de `φ`, pelo produto exterior que define `Bφ`. Assim, a verificação não repete apenas o mesmo caminho de código. Testar também `A = I` e o sinal da convenção escolhida.
2. **A rede recebe apenas a forma.** Usar `A`, seus valores singulares ou a métrica exata como atributos de entrada tornaria a tarefa diferente. Tais quantidades podem constar dos diagnósticos e da construção dos conjuntos.
3. **Escala não é ruído.** Padronização afim ajustada no treino pode ser invertível; dividir cada forma por sua própria norma remove informação exigida por `F(tφ) = t^(2/3) F(φ)`, salvo se a escala for preservada separadamente. O plano pede não descartá-la.
4. **Augmentation tem de transformar ambos os tensores.** Aplicar `C*` à forma e congruência à métrica. O uso de rotações no treinamento preserva a faixa dos valores singulares, enquanto transformações não ortogonais são diagnóstico separado.
5. **A distribuição fonte já possui simetria de rotação.** Isto é uma dedução do gerador do planejamento: se `A = Q₁ D Q₂`, o pullback por `C ∈ SO(7)` corresponde a `A C = Q₁ D (Q₂ C)`; pela invariância de Haar, `Q₂ C` tem a mesma distribuição que `Q₂`. Portanto, a augmentation pode melhorar o aproveitamento de uma amostra finita e a otimização, mas não amplia o suporte populacional dessa distribuição. Seu benefício não deve ser pressuposto.
6. **O cone positivo deve ser monitorado.** A saída triangular com diagonal estritamente positiva garante positividade em aritmética exata; condicionamento, finitude e autovalores continuam sendo verificações numéricas úteis. Para o ridge, registrar falhas sem ocultá-las.
7. **Generalização é mais de uma pergunta.** Distinguir dados novos na faixa de treino, extrapolação de escalas/anisotropias, rotações novas, mudanças de base não ortogonais e homogeneidade. Bons resultados em um desses eixos não resolvem os outros.
8. **A hipótese pode falhar.** Um resultado negativo, com orçamento comparável, várias sementes e diagnóstico claro, atende ao espírito da aula 13. O relatório deve relatar o que foi observado, sem declarar equivariância exata, descoberta de uma fórmula nova ou ganho de velocidade que não foi medido.

## Escopo das afirmações desta análise

As descrições acima registram o material local fornecido e sua relação com o planejamento. Elas não certificam todas as afirmações gerais dos slides como teoremas sem hipóteses. Em particular, garantias de aproximação, convergência, invariância em domínios discretos e ganhos empíricos de treinamento dependem das condições do problema. A implementação precisa apenas das fórmulas concretas do planejamento e de princípios experimentais verificáveis.
