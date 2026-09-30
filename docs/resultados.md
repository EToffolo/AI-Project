# Interpretação da execução completa

Resultados de [configs/full.json](../configs/full.json), sem ajuste posterior dos hiperparâmetros com base nos testes. [Relatório completo](../outputs/full/report.md), [métricas CSV](../outputs/full/metrics.csv), [metadados](../outputs/full/metadata.json).

Foram gerados 30.000 pares na distribuição de referência, divididos em 21.000/4.500/4.500 para treino/validação/teste, mais 4.500 pares independentes para extrapolação. Os modelos usam subconjuntos de 2.100, 7.000 e 21.000 exemplos de treino, com sementes 11, 22 e 33: nove ajustes de ridge e dezoito de redes. Todos os 27 modelos foram salvos e recarregados; as predições recarregadas concordaram com as registradas. Os hashes dos dados e do código foram conferidos.

## Precisão

Esta tabela apresenta a **média, entre as três sementes, das medianas e dos percentis 95**. Não são estatísticas calculadas reunindo todos os exemplos de todas as sementes. Todos os valores são erros relativos de Frobenius, em porcentagem, no maior conjunto de treino.

| Modelo | ID mediana | ID p95 | OOD mediana | OOD p95 |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 36,58% | 48,29% | 62,29% | 72,86% |
| MLP | 25,81% | 33,92% | 43,57% | 64,60% |
| MLP com rotações | 24,99% | 33,25% | 42,79% | 62,87% |

A rede aprende uma aproximação melhor que a referência linear. O aumento de dados reduz o erro mediano em cerca de 3,2% relativos em ID e 1,8% relativos em OOD em comparação com a rede comum. São ganhos modestos. Um erro de aproximadamente 25% em ID e 43% em OOD **não caracteriza alta precisão**.

As curvas por tamanho de treino mostram ganhos preditivos com mais exemplos. Nos experimentos menores o benefício da augmentação é maior. As redes com augmentação frequentemente usam quase todo o orçamento de 100 épocas; o experimento não prova convergência completa. Também não se escolheu retrospectivamente um orçamento maior por observar esses testes.

## Equivariância: um ganho e uma armadilha de interpretação

No maior conjunto, as médias das medianas do defeito SO(7) são:

| Modelo | ID | OOD |
| --- | ---: | ---: |
| Ridge | 0,01519 | 0,01323 |
| MLP | 0,23550 | 0,29848 |
| MLP com rotações | 0,22228 | 0,28175 |

A augmentação reduz esse defeito nas três sementes e nos dois domínios, sem piorar o erro mediano. Portanto, satisfaz o critério operacional declarado antes da execução. A redução média relativa é de aproximadamente 5,6%. Isso é evidência experimental sobre esses dados e orçamentos, não prova de equivariância ou significância estatística.

O ridge tem defeito de rotação muito menor que as redes, mas também erro de predição maior. A inspeção do modelo da semente 11 mostra que a predição média é muito próxima de `1,08288 I`; a variação relativa média das predições em torno dessa média é apenas cerca de 1,12%. Uma matriz escalar constante satisfaz exatamente a identidade de rotação e pode, ainda assim, ignorar quase toda a informação geométrica. Logo, o menor defeito do ridge **não indica que ele aprendeu melhor a aplicação F**.

Sob mudanças de base não ortogonais, o defeito mediano médio da MLP cai de 0,26582 para 0,25190 em ID e de 0,32437 para 0,31183 em OOD com augmentação. Esse diagnóstico é separado do critério principal por rotações; não há garantia arquitetural de naturalidade GL⁺(7).

## Escala e limites

O teste de homogeneidade distingue defeito da identidade e erro contra a métrica verdadeira escalada. As redes falham expressivamente em escalas distantes, especialmente ao multiplicar a forma por 4. O treinamento por rotações não ensina automaticamente o expoente 2/3 da lei de escala. Todas as configurações e fatores estão registrados, e o gráfico de escala apresenta cada fator separadamente.

Os resultados não permitem extrapolar para singularidades, condicionamentos arbitrários, variedades globais ou estruturas sem torção. A fórmula exterior exata continua sendo a referência: os testes de geração concordam com AᵀA dentro da precisão de ponto flutuante. O interesse aqui é estudar o aprendizado geométrico em um problema controlado.

## Reprodutibilidade e revisão

Os 24 testes cobrem álgebra exterior, composição, homogeneidade, positividade, Haar SO(7), splits, extrapolação, modelos, seleção por validação, augmentação correta e persistência. A execução rápida testa o fluxo completo. A execução integral produz relatório, gráficos, predições e checkpoints. Os quatro gráficos integrais foram inspecionados visualmente.

As verificações não substituem demonstrações ou revisão científica humana. A declaração da assistência de IA está em [uso-ia.md](uso-ia.md). Uma entrega acadêmica final deve revisar o texto, os resultados e a formatação exigida pela disciplina; o relatório gerado é uma base auditável em Markdown, não uma apresentação ou PDF diagramado de cinco páginas.
