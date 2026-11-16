# S.B.P. — Smart Bus Point · Inteligência Artificial & Machine Learning

**Global Solution 2026.2 — Desafio CitizenTech**
FIAP · Engenharia de Software · 3º ano de Fev

> Quanto tempo o cidadão espera no ponto de ônibus? Um modelo treinado com o GTFS oficial da SPTrans estima o intervalo de espera de linhas que ele nunca viu, e um agrupamento das 21.933 paradas de São Paulo indica onde instalar os totens primeiro.

---

## Ficha da CitizenTech

| Campo | Conteúdo |
| :--- | :--- |
| **Nome da CitizenTech** | S.B.P. — Smart Bus Point |
| **Cidade e bairro** | São Paulo (SP) — rede municipal de ônibus da SPTrans |
| **Cidadão beneficiado** | Passageiro que depende do ônibus e espera no ponto sem saber quanto tempo falta: trabalhador, estudante, idoso e pessoa com deficiência visual |
| **Problema, em uma frase** | O cidadão que espera no ponto não sabe quanto tempo ainda vai esperar, e essa incerteza o afasta do transporte público. |
| **Como funciona** | Um totem na calçada, um aplicativo e um site mostram a previsão de chegada, o mapa do trajeto e as referências urbanas, ligados por um QR Code. O totem funciona para quem não tem smartphone, bateria ou dados móveis. |
| **O que esta disciplina faz** | Transforma o problema em uma pergunta de ML e a responde com dado real e público: estima a espera onde não há dado publicado e define a fila de instalação dos totens |
| **Indicador de impacto** | Erro médio, em minutos, da espera estimada (**7,6 min**, contra 10,5 min da linha de base, e sem viés sistemático); nº de paradas de alta espera na fila de prioridade (**2.763**) |
| **ODS conectados** | ODS 9, ODS 10, ODS 11 |
| **Integrantes** | Alice Santos Bulhões (RM554499) · Eduardo Oliveira Cardoso Madid (RM556349) · Guilherme da Cunha Melo (RM555310) · Nicolas Haubricht Hainfellner (RM556259) |
| **Vídeo (até 5 min, não listado)** | `[PREENCHER: link do YouTube]` |

---

## O que está neste repositório

```
GS-Machine-Learning-2-26/
├── README.md                         este arquivo
├── RELATORIO_ML.md                   relatório da entrega, do problema à conclusão
├── prompts.md                        registro do uso de IA
└── ml/
    ├── sbp_previsao_espera.ipynb     notebook executado, com todas as saídas
    ├── requirements.txt              dependências, nas versões usadas
    ├── dados/
    │   ├── README.md                 fonte, link, período e licença dos dados
    │   └── gtfs/                     GTFS oficial da SPTrans (dados reais)
    ├── figuras/                      os 11 gráficos gerados pelo notebook
    └── artefatos/                    modelo treinado e CSVs para o backend
```

## Como executar

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

No Windows, ative com `.venv\Scripts\activate` em vez do comando acima.

```bash
pip install -r ml/requirements.txt
```

```bash
cd ml && jupyter lab sbp_previsao_espera.ipynb
```

O notebook já está **executado e com todas as saídas salvas** — basta abri-lo para ver os resultados. Para reexecutar do zero (cerca de 5 minutos, dominados pela busca de hiperparâmetros):

```bash
cd ml && jupyter nbconvert --to notebook --execute --inplace sbp_previsao_espera.ipynb
```

> Execute sempre a partir da pasta `ml/`: os caminhos dos dados são relativos a ela.

## Os dados

**Dados reais e públicos. Nada foi simulado.**

| Item | Descrição |
| :--- | :--- |
| **Fonte** | SPTrans — São Paulo Transporte S.A. |
| **Link** | https://www.sptrans.com.br/desenvolvedores/ |
| **Conjunto** | GTFS estático da rede municipal de ônibus |
| **Período de validade** | 01/10/2023 a 01/06/2027 |
| **Obtenção** | 14/11/2026 |

Detalhes de cada arquivo em [`ml/dados/README.md`](ml/dados/README.md). Nenhum dado pessoal é utilizado: o GTFS descreve a oferta de transporte, não passageiros.

## A pergunta de ML e a resposta

> **Dado o desenho de uma linha de ônibus e o horário do dia, qual é o intervalo médio de espera, em minutos, entre dois ônibus dessa linha?**

**Dataset:** 39.777 registros · 1.309 linhas de ônibus · 2.236 viagens.

**Divisão treino/teste agrupada por linha** (`GroupShuffleSplit` por `route_id`): as 262 linhas do teste **nunca** foram vistas no treino. Uma divisão aleatória colocaria a mesma linha dos dois lados e inflaria a métrica — o notebook mede esse vazamento na seção 8.2.

**Modelo:** XGBoost com `reg:squarederror`, escolhido após comparação com linha de base, Ridge e Random Forest. A função-objetivo foi trocada de propósito: o erro absoluto subestimava a espera em 2 minutos na média, e subestimar engana quem está no ponto. **18 combinações** de 3 hiperparâmetros, com `GroupKFold` de 5 dobras **aplicado somente ao treino**.

| Métrica | Treino | **Teste (linhas inéditas)** | Linha de base |
| :--- | ---: | ---: | ---: |
| MAE (min) | 4,31 | **7,59** | 10,48 |
| RMSE (min) | 6,54 | **10,64** | 15,29 |
| R² | 0,80 | **0,507** | −0,02 |
| Viés (min) | — | **−0,09** | — |

**Redução de 27,6% no erro sobre a linha de base**, sem viés sistemático. O totem também recebe uma **faixa de 80%**, com cobertura verificada de 79,8% por predição conformal.

### Até onde dá para chegar

A decomposição da variância mostra que **53,7% está no patamar de cada linha** e 46,3% no perfil horário. Oráculos delimitam o campo: o perfil horário por tipo de linha chega a R² 0,350, e conhecer o patamar verdadeiro de cada linha chegaria a 0,625. Nosso 0,507 recupera cerca de **58% do que havia para ganhar**.

O patamar de uma linha, porém, **não é inferível da rede**: a correlação entre a espera de uma linha e a das linhas que dividem suas paradas é de apenas +0,089. O que falta é demanda e frota alocada, que o GTFS não publica.

### O caso difícil

Bloqueando uma região inteira de São Paulo, o R² cai para cerca de **0,30**. Para uma linha nova numa cidade conhecida, 0,507 é o número honesto; para uma cidade sem dado nenhum, é 0,30.

### Onde instalar os totens — KMeans + PCA

As 21.933 paradas foram agrupadas por oferta e espera (PCA com 6 componentes, 93,1% da variância; KMeans com k = 4):

| Grupo | Paradas | Linhas/parada | Espera no pico | Prioridade |
| :--- | ---: | ---: | ---: | :--- |
| **Parada isolada de longa espera** | 2.763 | 1,90 | **36,6 min** | **1 — equidade** |
| Hub de integração | 3.915 | 11,57 | 20,6 min | 2 — alcance |
| Corredor de bairro | 7.156 | 3,53 | 17,1 min | 3 — expansão |
| Parada integrada ao metrô | 8.099 | 2,49 | 14,5 min | 4 — já bem servida |

## O que isso entrega ao cidadão

O modelo não substitui o dado publicado nem a telemetria em tempo real. O que ele resolve é o caso que a infraestrutura não cobre: a linha sem quadro de horários publicado e o momento em que a conexão cai. Aí o totem deixa de não ter informação nenhuma e passa a ter **a ordem de grandeza da espera** — saber que ali, àquela hora, se espera cerca de 10 minutos e não 40 muda a decisão de quem está na calçada. Para uma cidade sem GTFS algum o número cai para 0,30, e nesse caso o modelo serve de ponto de partida, não de resposta.

Leitura completa, com limites e próximos passos, em [`RELATORIO_ML.md`](RELATORIO_ML.md).

## Conexão com a CitizenTech

Esta disciplina gera três artefatos consumidos pelo restante do S.B.P.:

| Artefato | O que contém | Quem consome |
| :--- | :--- | :--- |
| `modelo_espera.joblib` | Modelo central, os dois modelos de faixa e a folga conformal | Quem for gerar novas estimativas |
| `espera_por_linha_hora.csv` | Espera estimada e faixa, por linha e hora | O painel do totem, que mostra a faixa ao cidadão |
| `prioridade_totens.csv` | As 21.933 paradas com grupo e posição na fila | A fila de instalação dos totens |

## Uso de IA

Conforme os combinados da GS, o uso de IA como copiloto está registrado em [`prompts.md`](prompts.md).
