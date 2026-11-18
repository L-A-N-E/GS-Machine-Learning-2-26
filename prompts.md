# Registro de uso de inteligência artificial generativa

Este arquivo registra o uso de IA no desenvolvimento da entrega de **Inteligência Artificial & Machine Learning** da Global Solution, conforme exigido pela atividade.

**Ferramenta:** Claude (Anthropic), via Claude Code.

A IA foi usada para construir o notebook a partir da pergunta de pesquisa definida pela equipe. Os prompts abaixo são reconstruções fiéis das instruções dadas ao longo do trabalho, e não transcrições literais. Todas as verificações descritas foram efetivamente executadas, e seus resultados estão nas saídas do notebook em `ml/sbp_previsao_espera.ipynb`.

---

## 1. Definição da pergunta de Machine Learning

### Prompt utilizado

> O problema do nosso projeto S.B.P. é que o cidadão espera no ponto de ônibus sem saber quanto tempo falta. Temos o GTFS da SPTrans. Transforme isso em uma pergunta de Machine Learning que seja genuinamente útil e que use dado real e público — e diga o que o modelo **não** vai conseguir resolver.

### Finalidade

Evitar um modelo que apenas reproduza o que o GTFS já informa. A pergunta precisava ter valor concreto para o cidadão e ser respondível com o dado disponível.

### Decisão da equipe

A equipe escolheu, entre três alternativas apresentadas, a combinação **tempo de espera (headway) mais agrupamento das paradas**, em vez de prever apenas o tempo de viagem entre paradas ou apenas o tempo de espera. O critério foi cobrir as duas perguntas que o projeto precisa responder: quanto o cidadão espera, e por onde começar a instalar os totens.

A equipe também definiu que o repositório seria `GS-Machine-Learning-2-26`, contendo apenas a análise e os artefatos que ela produz.

### Como a resposta foi verificada na prática

A pergunta foi conferida contra a rubrica da disciplina e contra o enunciado, em especial a exigência de dados reais e públicos. A origem dos dados foi documentada em `ml/dados/README.md` com órgão, link, período de validade (01/10/2023 a 01/06/2027, extraído dos campos `start_date` e `end_date` do próprio `calendar.txt`) e data de obtenção. **Nenhum dado foi simulado.**

---

## 2. Preparação dos dados e engenharia de atributos

### Prompt utilizado

> Monte o dataset a partir do GTFS: junte `frequencies`, `trips`, `routes`, `stop_times` e `stops`; crie atributos que descrevam a forma da linha — extensão por Haversine, número de paradas, duração, velocidade, sinuosidade, distância do centro, área de operação, linha noturna, integrações — e o momento do dia.

### Finalidade

O GTFS é relacional: o intervalo entre ônibus está em um arquivo e o que o explica está distribuído em outros. Era preciso reunir as tabelas e construir atributos que descrevessem a linha e o horário.

### Como a resposta foi verificada na prática

Antes de qualquer treino, os atributos foram conferidos contra o comportamento esperado da rede de São Paulo, e três padrões reais confirmaram a preparação:

- a espera mediana cai para 15 minutos nos horários de pico e **dobra na madrugada**, chegando a 30 minutos;
- as linhas noturnas do Corujão têm mediana de **60 minutos**, contra 15 das diurnas;
- a área Leste tem mediana de 12 minutos, enquanto Oeste, Sul e Nordeste ficam em 20.

Também foram conferidos os valores ausentes (17 em velocidade e 4.373 em sinuosidade, todos de viagens circulares em que origem e destino coincidem), a ausência de duplicatas e a ausência de valores fora da faixa plausível. O dataset final tem 39.777 registros, 1.309 linhas de ônibus e 2.236 viagens.

---

## 3. Divisão treino/teste e risco de vazamento

### Prompt utilizado

> Cada linha aparece cerca de 18 vezes no dataset, uma por faixa de hora, e quase todos os atributos são constantes dentro da linha. Analise o risco de vazamento em uma divisão aleatória e implemente a divisão correta, mantendo a aleatória para comparação no relatório.

### Finalidade

Esta é a decisão metodológica central do trabalho. Uma divisão aleatória colocaria a mesma linha nos dois lados, e a métrica resultante mediria memorização do patamar de cada linha, não generalização.

### Como a resposta foi verificada na prática

A divisão foi feita com `GroupShuffleSplit` agrupando por `route_id`, e o notebook **verifica explicitamente** que a interseção entre as linhas de treino e as de teste é zero, com a conferência impressa na saída da célula: 1.047 linhas no treino, 262 no teste, 0 em comum.

O efeito do vazamento foi medido e está documentado: o mesmo modelo, com os mesmos hiperparâmetros, avaliado com divisão aleatória, produz MAE de 5,28 minutos contra 7,59 da divisão correta, com 1.207 linhas presentes nos dois lados. **O número otimista seria falso**, e relatá-lo atribuiria ao projeto uma precisão que ele não entrega em campo.

---

## 4. Treino, busca de hiperparâmetros e avaliação

### Prompt utilizado

> Compare uma linha de base, Ridge, Random Forest e XGBoost na mesma divisão. Depois faça a busca em grade com 3 hiperparâmetros e pelo menos 10 combinações, com validação cruzada aplicada somente ao treino e mantendo o agrupamento por linha dentro das dobras.

### Finalidade

Garantir que a escolha do modelo viesse de evidência comparativa e que o conjunto de teste não participasse de nenhuma decisão de modelagem.

### Como a resposta foi verificada na prática

A comparação está registrada no notebook: linha de base 10,48, Ridge 8,40, Random Forest 8,18 e XGBoost 7,41 de MAE no teste. A busca percorreu **18 combinações** de `max_depth`, `learning_rate` e `n_estimators`, com `GroupKFold` de 5 dobras aplicado **somente ao treino**, totalizando 90 ajustes.

Foi conferido que a tabela das 18 combinações mostra o comportamento esperado de sobreajuste: as configurações mais profundas e com passo maior ajustam melhor o treino e pioram na validação. O modelo final ficou com MAE de 4,31 no treino e 7,59 no teste, R² de 0,507, contra 10,48 da linha de base — redução de 27,6% no erro.

---

## 5. Agrupamento das paradas

### Prompt utilizado

> Agrupe as paradas de São Paulo por oferta e espera com PCA + KMeans, para decidir onde instalar os totens primeiro. Nomeie cada grupo pelo perfil que ele realmente apresenta nos dados, e não pela ordem de espera.

### Finalidade

Responder a segunda pergunta do trabalho — por onde começar a implantação — e produzir um artefato consumível pelo backend do S.B.P.

### Problema encontrado e correção

A primeira versão nomeava os grupos pela ordem crescente de espera no pico. A conferência da tabela de perfis mostrou que os rótulos **não correspondiam aos dados**: o grupo rotulado como "bairro de espera moderada" tinha 11,57 linhas por parada e atendia 2,32 áreas de operação, ou seja, era o hub de integração. A nomeação foi refeita para identificar cada grupo pelo atributo em que ele é extremo, e os rótulos passaram a descrever o que os números mostram.

Na mesma revisão foi corrigida uma afirmação do relatório que os dados não sustentavam: o texto dizia que as paradas de maior espera formam um anel afastado do centro, quando as quatro distâncias médias ficam entre 10 e 17 km. O que separa os grupos é a **oferta de linhas**, não a distância do centro.

### Como a resposta foi verificada na prática

Foi conferido que a silhueta é modesta para todos os valores de k testados, entre 0,21 e 0,26, com máximo em k = 2 — as paradas formam um contínuo, e não grupos nitidamente separados. A escolha de k = 4 foi mantida por razão operacional e **essa limitação está declarada no relatório**, em vez de omitida.

O resultado foi exportado para `ml/artefatos/prioridade_totens.csv` com as 21.933 paradas classificadas, e a integração foi verificada de ponta a ponta: a classificação chega ao banco de dados do S.B.P. e aparece na resposta da API que alimenta o totem.

---

## 6. Rodada de melhoria do modelo

### Prompt utilizado

> O modelo está em R² 0,44 em linhas inéditas. Antes de tentar melhorar, descubra **onde está a variância do alvo** e qual é o teto do que dá para prever com este dado. Depois proponha e teste melhorias, e seja honesto sobre o que não funcionar.

### Finalidade

Evitar ampliar a grade de hiperparâmetros sem saber se havia margem a ganhar.

### O que a investigação mostrou

A decomposição da variância revelou que **53,7% está no patamar de cada linha** e 46,3% no perfil horário — a dificuldade está em inferir o patamar de uma linha nunca vista. Oráculos no conjunto de teste delimitaram o intervalo possível: o perfil horário por tipo de linha chega a R² 0,350, e conhecer o patamar verdadeiro chegaria a 0,625.

### O que funcionou

- **Densidade urbana da rede** (paradas e linhas num raio de 500 m e 1 km ao longo de todo o trajeto, via árvore k-d): único atributo novo com ganho real, de +0,045 de R² (0,462 para 0,507), medido por ablação no próprio notebook.
- **Troca da função-objetivo** para erro quadrático: o erro absoluto subestimava a espera em 2,03 minutos na média. A troca zerou o viés (−0,09) e ainda melhorou RMSE e R².
- **Predição conformal** para publicar uma faixa com cobertura verificada: 69,9% sem calibração, 79,8% depois dela.

### O que não funcionou, e está registrado no notebook

A hipótese de usar a espera das **linhas vizinhas** (que dividem as mesmas paradas) para estimar a de uma linha inédita. A correlação medida em 1.306 linhas foi de **+0,089**. Duas linhas que percorrem as mesmas ruas podem ter intervalos opostos, por serem uma troncal e outra alimentadora. O resultado negativo permaneceu no notebook porque explica o limite do problema.

### Uma correção de honestidade

A **validação espacial** (bloqueando uma região inteira de São Paulo por vez) mostrou que o R² cai de 0,507 para cerca de 0,30, revelando que a afirmação do relatório sobre servir a "cidades sem GTFS aberto" era mais forte do que o dado sustentava. O relatório passou a registrar os dois números.

### Como a resposta foi verificada na prática

Todas as métricas acima são saídas do notebook executado e podem ser reproduzidas rodando-o do início ao fim. O modelo final, os modelos de faixa e a folga conformal estão salvos em `ml/artefatos/modelo_espera.joblib`.
