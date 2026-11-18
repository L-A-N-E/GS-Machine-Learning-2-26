# Dados — origem, período e licença

Toda a análise do S.B.P. usa **dados reais e públicos**. **Nenhum dado foi simulado** e **nenhum dado pessoal é utilizado**: o GTFS descreve a oferta de transporte — linhas, paradas e horários —, não passageiros.

## Fonte

| Item | Descrição |
| :--- | :--- |
| **Órgão** | SPTrans — São Paulo Transporte S.A. (empresa municipal de São Paulo) |
| **Conjunto** | GTFS estático da rede municipal de ônibus |
| **Link** | https://www.sptrans.com.br/desenvolvedores/ |
| **Período de validade** | 01/10/2023 a 01/06/2027 (campos `start_date` e `end_date` de `calendar.txt`) |
| **Data de obtenção** | 14/11/2026 |
| **Licença** | Dado público, disponibilizado pela SPTrans para uso em aplicações de transporte |

## Arquivos em `gtfs/`

| Arquivo | Registros | Conteúdo | Papel na análise |
| :--- | ---: | :--- | :--- |
| `frequencies.txt` | 40.337 | Intervalo entre veículos (`headway_secs`) por viagem e faixa de hora | **Variável-alvo** |
| `trips.txt` | 2.268 | Viagens, com linha, sentido e tipo de dia | Chave de ligação |
| `routes.txt` | 1.365 | Linhas, nome, tipo e cor oficial | Atributos da linha |
| `stop_times.txt` | 99.120 | Horário programado de cada parada de cada viagem | Duração, nº de paradas |
| `stops.txt` | 22.273 | Paradas, com latitude e longitude | Geografia e extensão |
| `calendar.txt` | 6 | Dias da semana em que cada serviço opera | Tipo de dia |
| `agency.txt` | 1 | Operadora (SPTRANS) | Documentação |
| `fare_attributes.txt` / `fare_rules.txt` | 6 / 7.187 | Tarifas e regras de integração | Não usados nesta análise |

### Ausência de `shapes.txt`

O `shapes.txt` do GTFS original (60 MB) contém a geometria desenhada do traçado de cada rota. Ele **não é necessário** aqui — a extensão de cada linha é calculada pela soma das distâncias entre paradas consecutivas, por Haversine sobre as coordenadas de `stops.txt` — e foi removido para manter o repositório em tamanho razoável.

Para reproduzir com o arquivo completo, baixe o GTFS na área de desenvolvedores da SPTrans e descompacte em `gtfs/`.

## Variável-alvo

Para a maior parte da rede municipal, a SPTrans **não publica o horário de cada ônibus**: publica a **frequência**. Cada registro de `frequencies.txt` informa, para uma viagem e uma faixa de uma hora, de quantos em quantos segundos o veículo passa.

```
trip_id      start_time  end_time  headway_secs
1012-10-0    05:00:00    05:59:00  420            →  passa de 7 em 7 minutos
1012-10-0    01:00:00    01:59:00  3600           →  passa de 60 em 60 minutos
```

É a grandeza que o passageiro percebe no ponto. O alvo do modelo é, portanto, `headway_secs` convertido em minutos (`headway_min`), com valores entre 1 e 60 minutos e mediana de 20 minutos.

## Artefatos gerados

A execução do notebook grava em `../artefatos/`:

| Arquivo | Conteúdo |
| :--- | :--- |
| `modelo_espera.joblib` | Modelo treinado, para estimar a espera de linhas sem dado publicado |
| `espera_por_linha_hora.csv` | Espera publicada e estimada por linha e hora, consumida pelo backend |
| `prioridade_totens.csv` | As 21.933 paradas classificadas, com o grupo e a prioridade de instalação |

E as figuras do relatório em `../figuras/`.
