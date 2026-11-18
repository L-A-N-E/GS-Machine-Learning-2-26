# Relatório de Machine Learning — S.B.P. (Smart Bus Point)

**Global Solution 2026.2 · Desafio CitizenTech · Inteligência Artificial & Machine Learning**
FIAP — Engenharia de Software, 3º ano de Fev

**Integrantes:** Alice Santos Bulhões (RM554499) · Eduardo Oliveira Cardoso Madid (RM556349) · Guilherme da Cunha Melo (RM555310) · Nicolas Haubricht Hainfellner (RM556259)
**Repositório:** https://github.com/L-A-N-E/GS-Machine-Learning-2-26
**Vídeo (até 5 min, não listado):** https://youtu.be/0ISHAtgR2Fc

---

## 1. O problema e a pergunta de ML

O **S.B.P. (Smart Bus Point)** transforma a parada de ônibus em um ponto de informação acessível: um totem na calçada, um aplicativo e um site, ligados por um QR Code. O cidadão atendido é quem depende do ônibus — trabalhador, estudante, idoso, pessoa com deficiência visual — e chega ao ponto **sem saber quanto tempo ainda vai esperar**. Mais da metade dos passageiros aponta a espera como principal motivo de insatisfação, e o ônibus urbano perdeu mais de 44% dos pagantes em doze anos (NTU): a incerteza desloca demanda para o automóvel e para os aplicativos de transporte individual.

A maior parte desse problema se resolve com dado publicado e telemetria. Resta o caso que a infraestrutura não cobre — a parada sem dado publicado e a indisponibilidade de conexão. Daí a pergunta:

> **Dado o desenho de uma linha de ônibus — extensão, número de paradas, região, duração, integrações — e o horário do dia, qual é o intervalo médio de espera, em minutos, entre dois ônibus dessa linha?**

E uma segunda, não supervisionada, para apoiar a implantação:

> **Quais paradas de São Paulo se parecem em oferta e espera, e quais devem receber os primeiros totens?**

## 2. Os dados e a preparação

| Item | Descrição |
| :--- | :--- |
| **Fonte** | SPTrans — São Paulo Transporte S.A. |
| **Link** | https://www.sptrans.com.br/desenvolvedores/ |
| **Conjunto** | GTFS estático da rede municipal de ônibus |
| **Validade** | 01/10/2023 a 01/06/2027 (`calendar.txt`) |
| **Obtenção** | 14/11/2026 |

**Nenhum dado foi simulado e nenhum dado pessoal é usado.** O GTFS descreve a oferta — linhas, paradas e horários —, não passageiros.

O alvo vem de `frequencies.txt`. Para a maior parte da rede a SPTrans não publica o horário de cada ônibus, e sim a **frequência**: de quantos em quantos segundos o veículo passa, por faixa de uma hora. É a grandeza que o passageiro percebe no ponto.

As tabelas foram unidas por `trip_id` e `route_id`, e delas derivaram atributos que descrevem a **forma da linha** (extensão somando Haversine entre paradas consecutivas, nº de paradas, duração, velocidade, sinuosidade, distância do centro, área de operação, linha noturna, integrações) e o **momento do dia** (hora, também em seno e cosseno, dado o caráter cíclico da variável; período; tipo de dia). A esses somou-se a **densidade urbana da rede**: com uma árvore k-d sobre as 22.273 paradas, foram contadas quantas paradas e linhas existem num raio de 500 m e 1 km de cada ponto do trajeto — o indício indireto de demanda que o GTFS permite construir.

Foi mantido apenas `route_type = 3` (ônibus). Ausentes de velocidade (17) e sinuosidade (4.373, de viagens circulares) foram preenchidos pela mediana. Sem duplicatas e sem valores fora de faixa.

**Dataset final: 39.777 registros · 1.309 linhas · 2.236 viagens · 29 atributos.** A exploração confirma três padrões da rede: a espera mediana cai para 15 min nos picos e **dobra na madrugada**; a área Leste tem mediana de 12 min contra 20 de Oeste, Sul e Nordeste; e as **linhas noturnas do Corujão têm mediana de 60 min**, contra 15 das diurnas.

## 3. Divisão treino/teste

Cada linha aparece cerca de 18 vezes (uma por faixa de hora) e quase todos os atributos são constantes dentro dela. Uma divisão **aleatória** colocaria a mesma linha dos dois lados, e a métrica mediria memorização do patamar de cada linha, não generalização. A divisão é **agrupada por `route_id`** (`GroupShuffleSplit`, 80/20), e o notebook verifica e imprime a interseção.

| | Registros | Linhas |
| :--- | ---: | ---: |
| Treino | 31.923 | 1.047 |
| Teste | 7.854 | 262 |
| **Em comum** | — | **0** |

## 4. Teto do problema

Antes do treino, a variância do alvo foi decomposta: **53,7% está no patamar de cada linha** (118,2) e 46,3% no perfil horário dentro dela (102,0). A dificuldade não está em descrever a variação da espera ao longo do dia, mas em inferir **o patamar de uma linha nunca vista**. Oráculos delimitam o intervalo possível, todos avaliados no mesmo conjunto de teste:

| Previsor | R² | MAE |
| :--- | ---: | ---: |
| Chute: mediana do treino | −0,019 | 10,48 |
| Só o perfil horário da cidade | 0,215 | 9,67 |
| Perfil horário × tipo de linha | 0,350 | 8,72 |
| ORÁCULO: patamar verdadeiro da linha | 0,554 | 6,42 |
| ORÁCULO: patamar verdadeiro + perfil | **0,625** | 5,66 |

Um modelo sem acesso à identidade da linha precisa superar 0,350 e não pode alcançar 0,625.

## 5. Escolha do modelo e da função-objetivo

Quatro modelos na mesma divisão: linha de base (MAE 10,48), Ridge (8,40), Random Forest (8,18) e **XGBoost (7,41)**. O XGBoost vence e é três vezes mais rápido que a Random Forest, o que pesa porque seria treinado 90 vezes na busca.

A escolha da **função-objetivo** depende de qual erro é mais custoso para o cidadão, e não apenas da métrica agregada.

| Função-objetivo | MAE | RMSE | R² | **Viés** | Subestima |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Erro absoluto (mediana) | 7,47 | 11,12 | 0,46 | **+2,03** | 47,5% |
| Quantil 0,60 | 8,01 | 11,64 | 0,41 | +1,21 | 43,0% |
| Quantil 0,65 | 8,01 | 11,46 | 0,43 | +0,40 | 40,2% |
| **Erro quadrático (média)** | 7,59 | **10,64** | **0,51** | **−0,09** | 41,2% |

O erro absoluto minimiza o MAE, mas **subestima a espera em 2 minutos na média**, porque a distribuição é assimétrica à direita. Anunciar 10 minutos quando a espera é de 20 mantém o passageiro no ponto sob premissa falsa; o erro para cima apenas o surpreende. O erro quadrático zera o viés ao custo de 0,12 min de MAE, e melhora RMSE e R². É o adotado.

## 6. Treino com diferentes hiperparâmetros

Grade completa de **3 hiperparâmetros** e **18 combinações**: `max_depth` (3, 5, 7) × `learning_rate` (0,03 · 0,08 · 0,15) × `n_estimators` (300, 600). Validação cruzada com **`GroupKFold` de 5 dobras, somente no treino**, mantendo o agrupamento por linha dentro das dobras — sem isso o vazamento reapareceria pela validação. **O teste não participou de nenhuma decisão.** Foram 90 ajustes em 128 segundos, e a tabela completa no notebook mostra o sobreajuste esperado: as configurações mais profundas e com passo maior ajustam melhor o treino e pioram na validação.

**Melhor combinação:** `max_depth=7`, `learning_rate=0,03`, `n_estimators=300`.

## 7. Resultados

| Métrica | Treino | **Teste (linhas inéditas)** | Linha de base |
| :--- | ---: | ---: | ---: |
| MAE (min) | 4,31 | **7,59** | 10,48 |
| RMSE (min) | 6,54 | **10,64** | 15,29 |
| R² | 0,80 | **0,507** | −0,02 |
| Viés (min) | — | **−0,09** | — |

**Redução de 27,6% no erro sobre a linha de base.** 46,2% das previsões erram até 5 min e 75,9% até 10 min. O R² de 0,507 fica entre o teto do trivial (0,350) e o absoluto (0,625): o modelo recupera cerca de **58% da margem disponível**.

Os atributos mais usados são madrugada, operação aos domingos, linha noturna e prefixo N. O modelo aprendeu sobretudo o **regime de operação** — quando e que tipo de linha —, e muito menos a geometria do trajeto.

### 7.1 Limites medidos

Cinco experimentos delimitam o que o modelo não faz. Todos estão no notebook, inclusive o de resultado negativo.

| Experimento | O que mediu | Resultado |
| :--- | :--- | :--- |
| **Overfitting** | treino contra teste | R² 0,80 × **0,507**. O modelo memoriza parte do patamar das linhas de treino; a busca reduziu a distância, mas não poderia eliminá-la |
| **Vazamento** | o mesmo modelo com divisão aleatória | MAE **5,28** e R² 0,70, com **1.207 linhas nos dois lados**. O número otimista seria falso: mede a repetição do que já foi visto |
| **Ablação da densidade** | com e sem o bloco de densidade urbana | R² **0,462 → 0,507** (+0,045). Ganho real, mas modesto: indício indireto não substitui a variável ausente |
| **Linhas vizinhas** | correlação entre a espera de uma linha e a das que dividem suas paradas, em 1.306 linhas | **+0,089**, R² de 0,008. **Sem poder preditivo.** Duas linhas na mesma rua podem ter intervalos opostos, por serem uma troncal e outra alimentadora |
| **Validação espacial** | bloqueando uma região inteira de São Paulo por vez | R² cai para **0,26–0,34**. Parte do acerto vinha de conhecer outras linhas da mesma região |

Os dois últimos se explicam em conjunto: **a oferta não se propaga entre linhas vizinhas**, o que confirma por outro caminho o diagnóstico da seção 4. A validação espacial também corrige uma afirmação anterior do relatório: para *uma linha nova em uma cidade já mapeada*, 0,507 é a métrica aplicável; para *uma cidade sem dado algum*, 0,30. Os dois números ficam registrados.

## 8. Faixa de predição calibrada

Anunciar "14 minutos" atribui ao modelo uma precisão que ele não tem; anunciar **"entre 8 e 25 minutos"** preserva a utilidade da informação. Dois modelos de quantil fornecem os extremos, mas quantis estimados não garantem cobertura e precisam de calibração. Foi aplicada **predição conformal**, separando parte do treino para calibração, sempre agrupada por linha:

| | Cobertura | Largura mediana |
| :--- | ---: | ---: |
| Quantis crus | 69,9% | 19,5 min |
| **Conformalizado** | **79,8%** | 21,8 min |

Com uma folga de 1,14 min, a cobertura observada passa a corresponder à prometida. O totem anuncia a faixa e **a confiança associada a ela**.

## 9. Onde instalar os totens — KMeans + PCA

As 21.933 paradas com perfil completo foram descritas por 9 atributos de oferta e espera, padronizadas, reduzidas por **PCA** (6 componentes, 93,1% da variância) e agrupadas por **KMeans**. A silhueta é modesta para todo k (0,21–0,26, máximo em k=2): as paradas formam um **contínuo**, não grupos separados. Adotou-se k=4 por razão operacional — k=2 apenas separaria "muita" de "pouca" oferta — e os grupos são tratados como faixas de um espectro.

| Grupo | Paradas | Linhas/parada | Espera no pico | Prioridade |
| :--- | ---: | ---: | ---: | :--- |
| **Parada isolada de longa espera** | 2.763 | 1,90 | **36,6 min** | **1 — equidade** |
| Hub de integração | 3.915 | 11,57 | 20,6 min | 2 — alcance |
| Corredor de bairro | 7.156 | 3,53 | 17,1 min | 3 — expansão |
| Parada integrada ao metrô | 8.099 | 2,49 | 14,5 min | 4 — já bem servida |

O que separa os grupos **não é a distância do centro** — os quatro ficam entre 10 e 17 km —, mas a **oferta**: o hub concentra cerca de 11 linhas e atende mais de uma área, enquanto a parada isolada tem menos de 2 linhas e quase 37 min de espera no pico. Cabe a ressalva: o agrupamento mede a oferta, não **quantas pessoas** usam cada parada. Colocar equidade à frente de alcance foi uma escolha de política da equipe, não um resultado do dado.

## 10. Conclusão

O modelo erra cerca de **7,6 minutos** ao estimar a espera de uma linha que nunca viu, contra 10,5 da mediana da cidade, e **sem viés sistemático**. Não é precisão para anunciar "faltam 4 minutos" — essa informação continua vindo do dado publicado e da telemetria.

É precisão suficiente para **identificar o regime de espera**. Distinguir uma espera de cerca de 10 minutos de uma de 40 altera a decisão de quem está na calçada: esperar, caminhar até outro ponto ou procurar outro modal. A alternativa atual é a ausência de informação.

**Por que o trabalho para aqui.** As seções 4 e 7.1 respondem em conjunto. Mais da metade da variância está no patamar de cada linha, e esse patamar **não é inferível da estrutura da rede**. Não se trata de ajuste insuficiente nem de falta de um modelo mais sofisticado: trata-se de uma variável ausente no dado público — a demanda e a frota alocada àquela linha. Reconhecer o limite tem mais valor do que acrescentar atributos sem efeito no resultado.

**Impacto medido:** erro médio da espera estimada (**7,6 min**) e nº de paradas de alta espera na fila de prioridade 1 (**2.763**).

**Próximo passo.** O limite indica que dado buscar: (1) **API Olho Vivo** da SPTrans, que muda o alvo de espera programada para realizada; (2) **Censo 2022 do IBGE**, população e renda por setor censitário cruzados com a coordenada da parada, o melhor proxy público de demanda; (3) **bilhetagem agregada por linha**, se publicada — a medida direta da variável ausente.

---

### Conexão com os ODS

- **ODS 11 — Cidades e comunidades sustentáveis:** informação confiável na parada reduz a incerteza que desloca demanda para modais individuais.
- **ODS 10 — Redução das desigualdades:** a fila prioriza as paradas isoladas de longa espera, e o totem atende quem não tem smartphone, bateria ou dados móveis.
- **ODS 9 — Indústria, inovação e infraestrutura:** a solução opera fora do cenário ideal de conectividade, com dado aberto e modelo próprio.

### Artefatos para o restante da CitizenTech

| Artefato | O que contém | Quem consome |
| :--- | :--- | :--- |
| `modelo_espera.joblib` | Modelo central, os dois modelos de faixa e a folga conformal | Geração de novas estimativas |
| `espera_por_linha_hora.csv` | Espera estimada e faixa, por linha e hora | Painel do totem |
| `prioridade_totens.csv` | As 21.933 paradas com grupo e posição na fila | Fila de instalação dos totens |
