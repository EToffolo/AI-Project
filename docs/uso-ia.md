# Registro de assistência de IA

Este projeto foi implementado com assistência do Codex, a pedido do autor, a partir do planejamento fornecido. A assistência incluiu leitura das três páginas do planejamento, implementação, testes, documentação, execução dos experimentos e geração dos relatórios.

O trabalho foi dividido entre agentes para implementação matemática, modelos, documentação e revisão de testes. Essa divisão é uma revisão de software assistida por IA, não uma revisão científica independente por pesquisadores humanos.

As identidades matemáticas foram verificadas computacionalmente por rotas distintas: rótulos AᵀA versus cálculo exterior a partir apenas de φ; casos identidade/diagonal; naturalidade por matrizes não ortogonais; composição de pullbacks; homogeneidade; e compatibilidade NumPy/PyTorch. Esses testes não substituem demonstrações matemáticas. As convenções foram comparadas com a referência original de Grigorian.

Os números dos relatórios são calculados a partir das execuções, e não preenchidos manualmente. Os arquivos `config.json`, `metadata.json`, `dataset.json`, `metrics.csv`, `histories/` e os checkpoints permitem auditar configurações, versões, dados e resultados. Cada execução registra hashes SHA-256 dos módulos Python; uma mudança posterior no código não altera os metadados de uma execução anterior.

O autor deve revisar a interpretação e seguir a política da disciplina sobre declaração de assistência de IA antes de entregar o trabalho. Nenhum resultado é apresentado como descoberta de uma nova fórmula de métrica ou como prova de equivariância da rede.
