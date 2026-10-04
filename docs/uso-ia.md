# Registro de assistência de IA

Este projeto foi implementado com assistência do Codex, a pedido do autor, a partir do planejamento fornecido. A assistência incluiu leitura das três páginas do planejamento, implementação, testes, documentação, execução dos experimentos e geração dos relatórios.

O trabalho foi dividido entre agentes para implementação matemática, modelos, documentação e revisão de testes. Essa divisão é uma revisão de software assistida por IA, não uma revisão científica independente por pesquisadores humanos.

As identidades matemáticas foram verificadas computacionalmente por rotas distintas: rótulos AᵀA versus cálculo exterior a partir apenas de φ; casos identidade/diagonal; naturalidade por matrizes não ortogonais; composição de pullbacks; homogeneidade; e compatibilidade NumPy/PyTorch. Esses testes não substituem demonstrações matemáticas. As convenções foram comparadas com a referência original de Grigorian.

Os números dos relatórios são calculados a partir das execuções, e não preenchidos manualmente. Os arquivos `config.json`, `metadata.json`, `dataset.json`, `metrics.csv`, `histories/` e os checkpoints permitem auditar configurações, versões, dados e resultados. Cada execução registra hashes SHA-256 dos módulos Python; uma mudança posterior no código não altera os metadados de uma execução anterior.

O autor deve revisar a interpretação e seguir a política da disciplina sobre declaração de assistência de IA antes de entregar o trabalho. Nenhum resultado é apresentado como descoberta de uma nova fórmula de métrica ou como prova de equivariância da rede.

## Relatório final em inglês

O Codex também auxiliou a redação e a diagramação do relatório em LaTeX
`docs/relatorio/relatorio-projeto-g2.tex`, com cinco páginas de conteúdo principal
e referências na sexta página. As referências bibliográficas foram consultadas nas
fontes originais. As tabelas e os três gráficos vetoriais são gerados por
`scripts/build_report_assets.py` a partir da execução completa registrada.

Na preparação do relatório foram conferidos os hashes do código e dos dados,
recalculadas as estatísticas dos 54 registros de métricas a partir das predições
salvas e verificadas as predições recarregadas dos nove modelos do maior tamanho
de treino nos dois domínios. Os 24 testes e uma nova execução smoke passaram.
Essas verificações documentam a consistência computacional; a revisão acadêmica
e a responsabilidade pela entrega continuam sendo do autor.
