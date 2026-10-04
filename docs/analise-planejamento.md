# Planejamento e decisões de implementação

Fonte primária: [Planajamento de projeto.pdf](Planajamento%20de%20projeto.pdf), 3 páginas. O nome no disco contém `Planajamento`; o arquivo foi preservado. A leitura e a conferência visual das três páginas antecederam a implementação.

## Problema matemático

Estuda-se a aplicação **pontual e algébrica** que leva uma 3-forma positiva em R⁷ à sua métrica. Não se resolvem PDEs, não se impõe torção nula e não se tenta descobrir uma fórmula desconhecida.

Na orientação padrão, φ₀ = e¹²³ + e¹⁴⁵ + e¹⁶⁷ + e²⁴⁶ − e²⁵⁷ − e³⁴⁷ − e³⁵⁶. As entradas são os 35 coeficientes de φ nas triplas i < j < k em ordem lexicográfica, sem fator 3!. O código usa índices de 0 a 6; o texto matemático usa 1 a 7.

O alvo exato é gφ = (det Bφ)^(−1/9) Bφ, com (Bφ)ᵢⱼ vol₀ = (ιᵢφ) ∧ (ιⱼφ) ∧ φ / 6. O cálculo exterior recebe apenas φ e é independente da matriz geradora A. Foram conferidos os sinais e a convenção nas equações 2.1–2.7 de [Grigorian](https://arxiv.org/html/1108.2465#S2).

Para o pullback `(A*φ)(u,v,w) = φ(Au,Av,Aw)`, os rótulos são g(A*φ₀) = AᵀA. São testadas também a naturalidade g(C*φ) = Cᵀg(φ)C e a homogeneidade g(tφ) = t^(2/3)g(φ), para t > 0.

## Rastreabilidade dos requisitos

| Requisito do PDF | Implementação |
| --- | --- |
| 30.000 pares em precisão dupla | `data.generate_dataset`, `configs/full.json` |
| A = Q₁ diag(exp(s)) Q₂, Q independentes Haar SO(7) | QR gaussiana com correção dos sinais de R e do determinante |
| sᵢ uniformes em [−0,35; 0,35] | Parâmetro `data.log_bound` |
| Conferência pela fórmula exterior e identidade | Validação automática ao gerar e testes de álgebra |
| Rede recebe φ, nunca A | Arrays de entrada `(N,35)`; A não entra no treino |
| Split 70%/15%/15% antes de aumentar dados | IDs de fonte disjuntos, salvos no NPZ |
| Padronização somente no treino | Um `Standardizer` por subconjunto; compartilhado pelos três métodos |
| OOD: [−0,7; 0,7], max abs(s) > 0,35 | Rejeição antes de gerar pares OOD independentes |
| Ridge sobre 28 entradas | Tikhonov por mínimos quadrados aumentados, intercepto livre |
| MLP 35–128–128–28, ReLU | `models.MetricMLP` |
| Saída L Lᵀ; diagonal softplus + epsilon | Camada final estruturada, epsilon = 10⁻⁵ |
| Adam, early stopping pela validação | `training.train_mlp`, restaura menor loss de validação |
| Média do erro relativo de Frobenius ao quadrado | Conta todas as 49 entradas, inclusive as simétricas |
| Augmentação online SO(7) | Nova rotação por exemplo e minibatch, transforma φ e g antes de padronizar |
| Arquitetura, dados fonte e orçamento máximo iguais | Mesmo seed de inicialização/ordem dos minibatches; RNG de augmentação separado |
| Mediana e percentil 95 ID e OOD | `evaluation.evaluate` e tabelas CSV/JSON |
| Três seeds e curvas por tamanho de treino | Seeds 11, 22, 33; subconjuntos aninhados 2.100, 7.000, 21.000 |
| Equivariância rotações / GL⁺ separadas | Mesmas transformações inéditas para todos os modelos; métricas por domínio |
| Lei de escala | Fatores 0,25, 0,5, 2 e 4; defeito e erro de predição registrados separadamente |
| Falhas de positividade da regressão linear | Conta menor autovalor ≤ 0, também verifica MLP numericamente |
| Código, dados, gráficos e relatório | CLI, NPZ+manifesto, checkpoints, figuras PNG, relatório Markdown |
| Assistência de IA documentada | `docs/uso-ia.md` |

## Escolhas necessárias, não especificadas numericamente no PDF

O arquivo `configs/full.json` fixa **antes da execução**: learning rate 0,001; batches de 256; no máximo 100 épocas; paciência de 15 épocas; melhora mínima de 10⁻⁵ para reiniciar a paciência. O checkpoint sempre guarda a menor loss de validação, mesmo se a melhora for menor que esse limiar. Early stopping pode produzir números diferentes de atualizações efetivas; o orçamento máximo é igual.

A validação seleciona alpha entre 10⁻⁴, 10⁻², 1, 100 e 10.000. A convenção ridge usa a **soma** de quadrados, e não sua média. O critério de seleção de alpha é a mesma loss relativa usada na MLP. O conjunto OOD tem 4.500 amostras. Os diagnósticos geométricos usam 512 exemplos por domínio; a precisão preditiva usa todos os 4.500 exemplos de cada teste. As transformações não ortogonais usam |log σᵢ(C)| ≤ 0,2.

Os modelos são selecionados apenas pela validação ID. Não há ajuste posterior ao observar o teste/OOD. As execuções menores `smoke` servem apenas para testar o funcionamento do software, não para decidir hiperparâmetros científicos.

Para operacionalizar “sem perda material de precisão”, adotou-se tolerância de **5% de aumento relativo na mediana do erro** em cada domínio e semente no maior tamanho de treino. O relatório verifica se o defeito SO(7) cai em todas as sementes nos dois domínios e apresenta GL⁺ separadamente. Esse é um critério de implementação declarado, não um valor especificado no PDF. Três sementes avaliam variabilidade computacional, sem provar significância estatística nem propriedades universais.

## Interpretação correta

- Todas as formas positivas estão na mesma órbita GL⁺(7). A separação pertinente é por **amostra fonte antes da augmentação**, não por órbitas GL⁺.
- Como Q₂ é Haar, a distribuição fonte já é invariante a rotações à direita. A augmentação modifica a exposição finita e o treinamento, não o suporte da distribuição. Uma melhora não está garantida.
- Uma predição aproximadamente constante e isotrópica pode ter defeito de equivariância pequeno e erro preditivo grande. A homogeneidade e a precisão são necessárias para interpretar esse resultado.
- A saída de Cholesky garante positividade em aritmética exata, mas não equivariância. Falhas numéricas de positividade continuam sendo contadas.
- A padronização afim global é invertível e mantém informação de escala. Normalizar cada amostra para norma 1 destruiria essa informação.
- Rotações preservam o intervalo de valores singulares. Transformações GL⁺ e escala podem sair do suporte ID; os diagnósticos devem ser interpretados separadamente.
- O cálculo exato é o padrão de referência. O projeto não demonstra vantagem de velocidade ou substituição da fórmula conhecida.

## Escopo adotado

A implementação usa Python, NumPy, PyTorch, regressão regularizada, MLP, Adam, augmentação e um protocolo experimental reproduzível. O uso de Cholesky, álgebra exterior e geometria G₂ segue o planejamento. Arquiteturas adicionais não são necessárias à tarefa pontual definida no documento.

Referências do planejamento: [Karigiannis](https://arxiv.org/abs/1909.09717), [Grigorian](https://arxiv.org/abs/1108.2465), [Chen, Dobriban e Lee](https://jmlr.org/papers/v21/20-163.html). A documentação oficial de [reprodutibilidade do PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html) fundamenta o registro de versões e as limitações entre plataformas.
