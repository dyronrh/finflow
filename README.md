# Quant Portfolio Intelligence

Plataforma cuantitativa para analizar acciones, generar rankings multifactoriales, proponer composiciones de cartera, controlar riesgo, recomendar rebalanceos y producir señales explicables de **Long**, **Reduce**, **Avoid** y, en una fase posterior, **Short**.

> Estado: diseño de arquitectura / MVP en construcción  
> Horizonte inicial: acciones y ETFs de Estados Unidos, frecuencia diaria  
> Estrategia inicial: long-only multifactorial con rebalanceo quincenal o mensual  
> Principio central: primero construir un sistema reproducible y auditable; después incorporar ML, NLP, long-short y automatización de ejecución.

---

## 0. Estado actual del repositorio y quickstart

Implementado (núcleo del MVP, Sprints 2–5):

| Módulo | Contenido |
|---|---|
| `configs/strategies/v0.1.0.yaml` | Pesos del composite, umbrales de señales, límites de cartera y costes, validados con Pydantic (`quant_core/config.py`) |
| `packages/data_platform/` | Helpers point-in-time (`available_at <= as_of`), adaptadores Yahoo Finance / SEC EDGAR / S&P 500, mercado sintético, snapshots de features PIT, data contracts |
| `packages/quant_core/factors/` | Winsorización + percentil sectorial, scores por familia 0–100, composite con cobertura mínima |
| `packages/quant_core/signals/` | Filtros de elegibilidad, etiquetas `STRONG_LONG`…`AVOID`, explicaciones legibles |
| `packages/quant_core/portfolio/` | Selección con histéresis, equal weight / inverse volatility, límites por acción y sector, bandas de no-operación, límite de turnover |
| `packages/quant_core/execution/` | Modelo de costes (comisión + half-spread + slippage) |
| `packages/backtesting/` | Backtester event-driven: señal al cierre `t`, fill a la apertura `t+1`, costes, benchmarks (equal-weight y SPY), métricas, metadata reproducible; ajuste walk-forward con holdout e IC de factores |
| `apps/api/` | FastAPI: `GET /v1/rankings`, `GET /v1/securities/{id}/analysis`, `POST /v1/portfolios/{id}/rebalance/proposal` (solo propuesta, `PENDING_APPROVAL`) |
| `tests/` | Normalización, scoring, reglas de señales, leakage point-in-time, límites de cartera, no same-bar execution, determinismo, API |

**Datos.** Hay dos fuentes:

- `synthetic`: mercado sintético determinista para desarrollo y tests. Sus resultados no dicen nada sobre desempeño real.
- `real`: histórico de mercado real armado con fuentes gratuitas (`packages/data_platform/real_market.py`):

| Dato | Fuente | Point-in-time |
|---|---|---|
| Precios diarios y SPY | Yahoo Finance (`yfinance`) | Sí: se usa el cierre de `t` y se ejecuta en la apertura de `t+1`. Los retornos usan precios ajustados por dividendos; market cap, ADV y filtro de precio usan el precio negociado (ajustado solo por splits), porque el ajuste por dividendos incorpora dividendos futuros |
| Fundamentales trimestrales y acciones en circulación | SEC EDGAR XBRL (`companyfacts`) | Sí: `available_at` = fecha de filing; se conserva el **primer** valor publicado, nunca reexpresiones |
| Universo y sector | S&P 500 en Wikipedia (miembros actuales + historial de altas/bajas) | Aproximado (ver limitaciones) |
| Estimaciones de analistas | — | No hay fuente gratuita PIT: el factor *revisions* queda desactivado y su peso se redistribuye |

Yahoo no se usa para fundamentales porque solo entrega los últimos trimestres y sin fecha de publicación.

```bash
uv sync --extra real-data
export SEC_USER_AGENT="finflow research tu-email@ejemplo.com"   # exigido por la SEC
make fetch-data          # descarga y cachea en data/local_dev_only/real/ (~30–60 min la primera vez)
make backtest-real       # backtest v0.1.0 sobre histórico real
make tune-real           # ajuste walk-forward + holdout + IC de factores
make tune-real-weights   # igual, pero probando perfiles de pesos de factores
```

**Protocolo de ajuste** (`packages/backtesting/tuning.py`, `pipelines/tune_strategy.py`):

1. **Holdout:** los últimos 3 años (por defecto) nunca se usan para elegir parámetros y se miran una sola vez.
2. **Grilla:** cada configuración se corre una vez. La grilla por defecto cubre el límite de turnover, el umbral de LONG y el tipo de ponderación; con `--grid archivo.yaml` se define otra.
3. **Walk-forward:** en cada ventana de 3 meses se elige la mejor configuración con los 5 años previos y se mide su resultado fuera de muestra. Esto estima cuánto vale el *proceso de ajuste*, no la mejor corrida.
4. **Validación final:** la mejor configuración pre-holdout se compara con v0.1.0 y el benchmark en el holdout.
5. **Diagnóstico:** IC de Spearman por familia, por variable individual y por año, más el spread de quintiles, para separar "los factores no predicen" de "la cartera no aprovecha la señal".

El resultado es un `candidate.yaml`, no una estrategia aprobada: adoptarlo como nueva versión requiere revisión humana (§8.3).

**Limitaciones del histórico gratuito:**
- **Sesgo de supervivencia residual:** las empresas que salieron del S&P 500 y ya no tienen historial en Yahoo quedan fuera, lo que favorece al backtest.
- **Sectores:** se usa la clasificación GICS actual, y el código SIC para las empresas retiradas.
- **Conceptos XBRL:** son aproximaciones (EBITDA = resultado operativo + D&A; deuda = deuda de largo plazo + corto plazo). Bancos y aseguradoras quedan con cobertura parcial.
- **Cambio de configuración:** el walk-forward no cobra el coste de cambiar de configuración entre ventanas.

Pendiente: fuente PIT de estimaciones de analistas, universo PIT completo (proveedor con delistings), PostgreSQL/DuckDB/MinIO, Docker Compose, frontend Next.js, risk engine (VaR/CVaR, beta, alertas), walk-forward, optimización convexa, paper trading.

---

## 1. Objetivo

Construir una plataforma propia inspirada en la lógica de productos quantamentales como Seeking Alpha:

1. Definir un universo invertible y líquido.
2. Obtener datos históricos y fundamentales con disciplina `point-in-time`.
3. Construir factores de inversión interpretables.
4. Rankear activos de forma relativa contra sus pares sectoriales.
5. Convertir el ranking en señales de cartera y pesos objetivo.
6. Incorporar costes de transacción, riesgo, liquidez y límites de concentración.
7. Generar sugerencias explicables de compra, reducción, salida y rebalanceo.
8. Validar todo con backtesting walk-forward fuera de muestra.
9. Hacer paper trading antes de cualquier integración de ejecución real.

La plataforma **no predice precios con certeza** ni debe presentar resultados pasados como promesas de rentabilidad. Su objetivo es sistematizar decisiones de selección, sizing, riesgo y ejecución.

---

## 2. Principios no negociables

### 2.1 Point-in-time data

Todo dato usado para generar una señal en la fecha `t` debe haber estado disponible antes del momento de decisión.

Ejemplos de errores prohibidos:

- Usar estados financieros publicados semanas después como si hubieran estado disponibles al cierre del trimestre.
- Usar componentes actuales del S&P 500 para simular una estrategia desde 2010.
- Ajustar el universo usando información futura sobre quiebras, delistings o adquisiciones.
- Ejecutar al cierre usando una señal calculada con precios de ese mismo cierre.

Cada observación debe preservar:

```text
event_time: cuándo ocurrió el evento económico
available_at: cuándo el dato estuvo disponible para el mercado o proveedor
ingested_at: cuándo fue descargado por nuestra plataforma
as_of_date: fecha de la decisión simulada o real
```

### 2.2 Separación entre investigación y producción

Nunca se debe permitir que un notebook experimental pase directamente a producción.

```text
research code
      ↓
validación walk-forward
      ↓
estrategia versionada
      ↓
paper trading
      ↓
aprobación humana
      ↓
producción / ejecución controlada
```

### 2.3 Trazabilidad total

Cada recomendación debe poder responder:

- Qué versión de datos la generó.
- Qué versión de factores y modelo se utilizó.
- Qué reglas de riesgo se aplicaron.
- Qué restricciones bloquearon o modificaron la orden.
- Qué precio, coste, spread y slippage se asumieron.
- Cuál era la cartera antes y después del rebalanceo.

### 2.4 Explicabilidad primero

La primera versión debe usar factores interpretables y reglas explícitas:

- Value.
- Growth.
- Profitability / Quality.
- Momentum.
- Earnings revisions.
- Riesgo, liquidez y concentración.

Los modelos ML deben ser complementarios, no el núcleo inicial de decisión.

---

## 3. Alcance funcional

### 3.1 MVP

El MVP debe entregar:

- Universo de acciones líquidas de Estados Unidos.
- Ranking diario de activos con score 0–100.
- Scores por factor: value, growth, profitability, momentum y revisions.
- Comparación de cada acción contra sector e industria.
- Filtros de liquidez, precio, market cap y calidad de datos.
- Etiquetas operativas:
  - `STRONG_LONG`
  - `LONG`
  - `WATCH`
  - `NEUTRAL`
  - `REDUCE`
  - `AVOID`
- Cartera modelo long-only de 20–30 activos.
- Pesos objetivo con límites por acción, sector y riesgo.
- Rebalanceo quincenal o mensual con bandas de no-operación.
- Backtest con costes de transacción, turnover, slippage y benchmarks.
- Dashboard con explicación de cada señal.
- Alertas de cambios relevantes de score, riesgo y cartera.

### 3.2 Fase posterior

- Exposición long-short market-neutral.
- Datos de borrow y coste de préstamo para posiciones short.
- Clasificación de regímenes de mercado.
- NLP de earnings calls, filings y guidance.
- Modelos de ranking con LightGBM/CatBoost.
- Optimización robusta con CVaR.
- Ejecución mediante brokers.
- Agentes de IA solo para investigación, explicaciones y monitoreo; no para ejecutar operaciones sin reglas duras.

---

## 4. Arquitectura objetivo

```text
                        ┌──────────────────────────────┐
                        │           Frontend           │
                        │ Next.js + TypeScript + React │
                        └──────────────┬───────────────┘
                                       │ HTTPS / REST
                        ┌──────────────▼───────────────┐
                        │          API Gateway         │
                        │      FastAPI + Pydantic      │
                        └──────────────┬───────────────┘
                                       │
       ┌───────────────────────────────┼────────────────────────────────┐
       │                               │                                │
┌──────▼────────┐             ┌────────▼─────────┐             ┌────────▼─────────┐
│ Research API  │             │ Portfolio API    │             │ Reporting API    │
│ rankings      │             │ positions        │             │ dashboards       │
│ factors       │             │ rebalancing      │             │ exports          │
│ backtests     │             │ constraints      │             │ audit logs       │
└──────┬────────┘             └────────┬─────────┘             └────────┬─────────┘
       │                               │                                │
       └──────────────────────┬────────┴────────────────────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │        Quant Decision Layer     │
              │ factor scoring                  │
              │ eligibility rules               │
              │ regime filters                  │
              │ portfolio optimizer             │
              │ transaction-cost model          │
              └───────────────┬────────────────┘
                              │
      ┌───────────────────────┼─────────────────────────┐
      │                       │                         │
┌─────▼─────────┐     ┌───────▼──────────┐     ┌────────▼──────────┐
│ Feature Store │     │ Backtest Engine  │     │ Risk Engine       │
│ point-in-time │     │ event-driven     │     │ exposure / CVaR   │
│ factors       │     │ walk-forward     │     │ limits / alerts   │
└─────┬─────────┘     └───────┬──────────┘     └────────┬──────────┘
      │                       │                         │
      └───────────────────────┼─────────────────────────┘
                              │
               ┌──────────────▼──────────────┐
               │       Data Platform          │
               │ raw → validated → curated   │
               │ → features → snapshots      │
               └──────────────┬──────────────┘
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
┌─────────▼────────┐ ┌────────▼─────────┐ ┌────────▼────────────┐
│ Market data      │ │ Fundamental data  │ │ Analyst estimates   │
│ OHLCV, volume    │ │ filings, ratios   │ │ EPS revisions       │
└──────────────────┘ └──────────────────┘ └─────────────────────┘
```

---

## 5. Lenguajes y tecnologías

### 5.1 Decisión principal

| Área | Lenguaje / tecnología | Razón |
|---|---|---|
| Investigación cuantitativa | Python | Ecosistema dominante para datos, ML, optimización, series temporales y backtesting |
| APIs backend | Python + FastAPI | Comparte modelos y lógica con el research layer; tipado y alto rendimiento para una API de producto |
| Frontend | TypeScript + Next.js + React | Interfaces ricas, rápidas y mantenibles para rankings, dashboards y gráficos |
| Consultas analíticas | SQL + DuckDB + PostgreSQL | SQL es esencial para investigación, snapshots point-in-time y consultas operativas |
| Jobs de datos | Python | Homogeneidad con el motor cuantitativo y facilidad para integraciones |
| Infraestructura | Docker + Terraform | Reproducibilidad y despliegue consistente |
| CI/CD | GitHub Actions | Pruebas, versionado, linting y despliegues automatizados |
| Observabilidad | OpenTelemetry + Prometheus + Grafana | Métricas, trazas y alertas del sistema |
| Caché / colas | Redis | Caché de rankings, rate limiting y trabajos asíncronos |
| Orquestación | Prefect o Dagster | Pipelines de datos con retries, observabilidad y dependencias explícitas |

### 5.2 Por qué Python es el lenguaje principal

Python debe ser el núcleo para:

- Ingesta y validación de datos.
- Ingeniería de features.
- Scoring multifactorial.
- Modelos estadísticos y de machine learning.
- Backtesting.
- Optimización de cartera.
- API de señales y portfolio intelligence.
- Monitoreo de modelos.
- Orquestación de jobs.

Librerías recomendadas:

```text
Core data:
- Python 3.12+
- Polars
- pandas
- NumPy
- PyArrow
- DuckDB

Stats / ML:
- scipy
- statsmodels
- scikit-learn
- LightGBM
- CatBoost
- PyTorch, solo cuando se justifique

Portfolio optimization:
- CVXPY
- cvxportfolio
- skfolio
- PyPortfolioOpt
- Riskfolio-Lib

Backtesting:
- vectorbt para investigación rápida
- motor propio event-driven para validación realista
- cvxportfolio para simulación de políticas con costes y restricciones

Backend:
- FastAPI
- Pydantic
- SQLAlchemy
- Alembic
- Celery o Dramatiq
- Redis

Testing:
- pytest
- hypothesis
- ruff
- mypy
- pre-commit
```

CVXPY es especialmente adecuado para expresar optimizaciones de cartera con restricciones, penalizaciones de riesgo y costes de turnover. Permite modelar problemas convexos directamente desde Python y admite programación parametrizada, útil cuando recalculas pesos con datos nuevos. [25]

Cvxportfolio puede usarse como acelerador en investigación porque ofrece simulación de mercado, políticas de optimización mono y multiperiodo, costes y restricciones personalizables, además de cachear datos y cálculos costosos como matrices de covarianza. [29]

### 5.3 Cuándo usar Rust, Go o C++

No los uses en la primera versión.

Úsalos solo si aparece una necesidad demostrable:

- **Rust:** pipeline de datos de muy alto rendimiento, cálculo de features sobre millones de observaciones o servicios críticos de ejecución con garantías de memoria.
- **Go:** microservicios concurrentes, websockets de mercado y gateways operacionales.
- **C++:** estrategias intradía de baja latencia o conectividad de ejecución de nivel institucional.

Para una plataforma diaria/semanal de ranking y portafolio, Python + Polars + DuckDB + PostgreSQL entrega rendimiento más que suficiente y permite iterar mucho más rápido.

---

## 6. Estructura del repositorio

```text
quant-portfolio-intelligence/
├── README.md
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── Makefile
├── .env.example
├── docs/
│   ├── architecture.md
│   ├── data-contracts.md
│   ├── factor-methodology.md
│   ├── risk-policy.md
│   ├── backtesting-policy.md
│   ├── execution-policy.md
│   └── adr/
│       ├── 001-python-fastapi.md
│       ├── 002-point-in-time-data.md
│       └── 003-event-driven-backtesting.md
├── apps/
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── schemas/
│   │   ├── dependencies/
│   │   └── services/
│   ├── web/
│   │   ├── app/
│   │   ├── components/
│   │   ├── lib/
│   │   └── tests/
│   └── worker/
│       ├── main.py
│       └── tasks/
├── packages/
│   ├── quant_core/
│   │   ├── domain/
│   │   ├── factors/
│   │   ├── signals/
│   │   ├── portfolio/
│   │   ├── risk/
│   │   ├── execution/
│   │   └── models/
│   ├── data_platform/
│   │   ├── ingestion/
│   │   ├── validation/
│   │   ├── transforms/
│   │   ├── point_in_time/
│   │   ├── storage/
│   │   └── contracts/
│   ├── backtesting/
│   │   ├── engine/
│   │   ├── costs/
│   │   ├── benchmarks/
│   │   ├── metrics/
│   │   └── reports/
│   ├── ml/
│   │   ├── features/
│   │   ├── training/
│   │   ├── validation/
│   │   ├── registry/
│   │   └── monitoring/
│   └── shared/
│       ├── config/
│       ├── logging/
│       ├── observability/
│       └── types/
├── pipelines/
│   ├── daily_ingestion.py
│   ├── fundamentals_update.py
│   ├── compute_features.py
│   ├── score_universe.py
│   ├── rebalance_portfolio.py
│   ├── run_backtest.py
│   └── generate_reports.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   ├── property/
│   └── fixtures/
├── notebooks/
│   ├── exploratory/
│   └── archived/
├── infra/
│   ├── docker/
│   ├── terraform/
│   ├── kubernetes/
│   └── monitoring/
└── data/
    └── local_dev_only/
```

Regla: los notebooks sirven para explorar; la lógica validada debe vivir en `packages/` con pruebas automatizadas.

---

## 7. Capas de datos

### 7.1 Bronze: datos crudos e inmutables

Guardar exactamente lo recibido desde proveedores:

```text
raw_prices/
raw_fundamentals/
raw_estimates/
raw_corporate_actions/
raw_macro/
raw_borrow/
raw_news/
```

Requisitos:

- Nunca sobrescribir registros originales.
- Guardar fuente, fecha de descarga, hash y versión de API.
- Guardar en Parquet particionado por fecha y proveedor.
- Mantener `ingested_at` y metadatos de licencia.
- Versionar cambios de schema.

### 7.2 Silver: datos limpios y normalizados

Transformaciones:

- Tickers normalizados.
- ISIN / CUSIP / FIGI cuando estén disponibles.
- Zona horaria estandarizada.
- Moneda normalizada.
- Ajustes por splits y dividendos.
- Eliminación y marcado de duplicados.
- Validación de precios imposibles, valores nulos y outliers.
- Histórico de cambios de ticker y corporate actions.

### 7.3 Gold: datasets analíticos

Tablas listas para modelos y producto:

```text
security_master
universe_membership_pit
daily_prices_pit
fundamentals_pit
analyst_estimates_pit
factor_features_daily
factor_scores_daily
portfolio_targets
portfolio_holdings
trades_simulated
backtest_results
risk_metrics_daily
signal_audit_log
```

### 7.4 Data contracts

Cada tabla crítica debe tener:

```yaml
dataset: factor_scores_daily
primary_key:
  - as_of_date
  - security_id
required_columns:
  - as_of_date
  - security_id
  - sector_id
  - composite_score
  - value_score
  - growth_score
  - profitability_score
  - momentum_score
  - revisions_score
  - data_version
  - model_version
quality_rules:
  - composite_score between 0 and 100
  - no duplicate primary keys
  - available_at <= as_of_date
  - sector_id is not null
```

---

## 8. Modelo de factores

### 8.1 Familias de factores

```text
Value
- earnings yield
- free cash flow yield
- EV/EBITDA
- EV/EBIT
- price-to-book
- shareholder yield

Growth
- revenue growth
- EPS growth
- EBITDA growth
- free cash flow growth
- margin expansion

Profitability / Quality
- ROIC
- ROE
- gross margin
- operating margin
- free cash flow margin
- net debt / EBITDA
- interest coverage
- accruals

Momentum
- 1, 3, 6 y 12 meses
- retorno residual vs mercado y sector
- distancia a SMA 50 / 200
- fuerza relativa
- volatilidad ajustada

Earnings revisions
- cambio de EPS estimado a 7, 30, 60 y 90 días
- porcentaje de revisiones al alza vs baja
- earnings surprise
- cambios de guidance
- dispersión de estimaciones

Risk / Liquidity
- volatilidad realizada
- beta
- máximo drawdown
- ADV
- spread estimado
- concentración sectorial
- riesgo de evento
```

### 8.2 Normalización

Los factores deben normalizarse por sector o industria.

```python
def winsorized_sector_percentile(
    dataframe,
    feature_column,
    sector_column="sector_id",
    lower_quantile=0.01,
    upper_quantile=0.99,
):
    ...
```

Proceso:

1. Aplicar winsorización para limitar outliers.
2. Invertir signos donde menor valor sea mejor.
3. Calcular percentil dentro de sector/industria.
4. Aplicar controles de cobertura y calidad de datos.
5. Construir score de factor entre 0 y 100.
6. Combinar factores con pesos versionados.

### 8.3 Score compuesto

```python
COMPOSITE_WEIGHTS = {
    "value": 0.25,
    "growth": 0.15,
    "profitability": 0.25,
    "momentum": 0.20,
    "revisions": 0.15,
}
```

```python
composite_score = (
    value_score * 0.25
    + growth_score * 0.15
    + profitability_score * 0.25
    + momentum_score * 0.20
    + revisions_score * 0.15
)
```

La versión inicial debe mantener pesos fijos y explícitos. Cualquier cambio requiere:

- Nueva versión de estrategia.
- Backtest reproducible.
- Comparación contra versión anterior.
- Revisión humana.
- Registro en `strategy_registry`.

---

## 9. Motor de señales

### 9.1 Señales Long

```python
long_eligible = (
    (composite_percentile >= 0.85)
    & (profitability_percentile >= 0.60)
    & (momentum_percentile >= 0.50)
    & (revisions_percentile >= 0.50)
    & (liquidity_percentile >= 0.60)
    & (risk_flag == 0)
)
```

### 9.2 Señales de reducción

```python
reduce_signal = (
    (composite_percentile <= 0.55)
    | (revisions_percentile <= 0.30)
    | (portfolio_risk_contribution > risk_budget)
    | (position_weight > max_position_weight)
)
```

### 9.3 Señales Short

Los shorts no se habilitan en el MVP.

Cuando se habiliten, deben requerir:

```python
short_eligible = (
    (composite_percentile <= 0.10)
    & (momentum_percentile <= 0.25)
    & (revisions_percentile <= 0.25)
    & (profitability_percentile <= 0.35)
    & (borrow_available is True)
    & (borrow_cost_annual <= max_borrow_cost)
    & (liquidity_percentile >= 0.70)
    & (event_risk_flag == 0)
)
```

Un ranking bajo por sí solo no es una señal short. Se requiere liquidez, borrow, coste aceptable, deterioro persistente y control de eventos.

---

## 10. Construcción de cartera

### 10.1 Orden de implementación

1. Equal weight.
2. Inverse volatility.
3. Risk parity.
4. Hierarchical Risk Parity.
5. Optimización convexa con restricciones.
6. CVaR / robust optimization.
7. Long-short market-neutral.
8. Reinforcement learning experimental.

### 10.2 Optimización base

```text
Maximizar:
retorno esperado - penalización de riesgo - coste de turnover

Sujeto a:
- suma de pesos = 1
- 0 <= peso por activo <= peso máximo
- exposición por sector <= límite sectorial
- exposición a activos ilíquidos <= límite
- turnover <= límite por rebalanceo
- contribución de riesgo <= presupuesto definido
```

Forma matemática:

\[
\max_w
\left(
\hat{\mu}^{T} w
-
\lambda w^{T}\Sigma w
-
\kappa \sum_i c_i |w_i - w_i^{prev}|
\right)
\]

Sujeto a:

\[
\sum_i w_i = 1
\]

\[
0 \leq w_i \leq w_{\max}
\]

\[
\sum_{i \in s} w_i \leq S_{\max}
\]

\[
\sum_i |w_i - w_i^{prev}| \leq T_{\max}
\]

La formulación media-varianza penaliza la rentabilidad esperada por el riesgo de la matriz de covarianza. Los ejemplos de CVXPY muestran precisamente este trade-off riesgo-retorno, con restricciones como suma de pesos igual a uno y long-only. También señalan la ventaja computacional de usar modelos de riesgo por factores frente a matrices de covarianza completas cuando crece el universo. [31]

### 10.3 Restricciones iniciales

```yaml
portfolio:
  max_position_weight: 0.08
  max_sector_weight: 0.25
  max_industry_weight: 0.15
  min_holdings: 20
  max_holdings: 35
  max_turnover_per_rebalance: 0.25
  min_adv_usd: 5000000
  min_market_cap_usd: 2000000000
  minimum_stock_price_usd: 5
  max_single_name_risk_contribution: 0.10
  rebalance_frequency: monthly
```

---

## 11. Rebalanceo y ejecución

### 11.1 Separar ranking de negociación

El score puede actualizarse diariamente; la cartera no debe rotar completamente cada día.

```text
Daily:
- actualizar datos
- recalcular factores
- recalcular scores
- detectar cambios críticos
- generar alertas

Weekly:
- revisar nuevos candidatos
- revisar exclusiones
- actualizar riesgo y exposición

Monthly / Biweekly:
- calcular pesos objetivo
- aplicar bandas de no-operación
- generar propuesta de rebalanceo
- aprobar o rechazar operación
```

### 11.2 Bandas de no-operación

Evitar transacciones pequeñas que no compensen costes:

```python
should_trade = abs(target_weight - current_weight) > trade_band
```

La banda debe depender de:

- Coste estimado.
- Liquidez.
- Ventaja esperada del cambio de score.
- Volatilidad.
- Tamaño actual de la posición.
- Cercanía de earnings u otro evento.

### 11.3 Modelo de costes

Cada backtest y simulación debe modelar:

```text
commission_cost
bid_ask_spread_cost
slippage_cost
market_impact_cost
borrow_cost
financing_cost
corporate_action_cost
tax_placeholder
```

Modelo inicial conservador:

```python
transaction_cost = (
    commission
    + notional * half_spread_bps / 10_000
    + notional * slippage_bps / 10_000
)
```

El modelo debe soportar execution timing explícito:

```text
signal generated: close of day t
order submitted: after market close t
fill assumption: next open, next VWAP, or next close at t+1
```

Nunca usar señal y fill al mismo cierre salvo que el feed, infraestructura y lógica justifiquen de manera realista esa posibilidad.

---

## 12. Backtesting riguroso

### 12.1 Requisitos

El backtest debe ser event-driven y reproducible.

```text
Input:
- universo point-in-time
- precios point-in-time
- fundamentales point-in-time
- estimaciones point-in-time
- reglas de factores versionadas
- costes de ejecución
- restricciones de cartera
- calendario de rebalanceo

Output:
- equity curve
- holdings históricos
- órdenes y fills simulados
- turnover
- costes
- exposición sectorial
- exposición por factor
- drawdowns
- resultados por régimen
- reporte HTML / JSON / Parquet
```

### 12.2 Validación temporal

No utilizar random train-test split para datos financieros temporales.

Usar walk-forward:

```text
Train window: 5 años
Validation window: 1 año
Test window: 3 meses
Step: 3 meses

Repetir hasta recorrer todo el histórico.
```

El análisis académico reciente que compara validaciones financieras remarca la diferencia entre divisiones aleatorias convencionales y validación temporal walk-forward; en series financieras, la segunda es la forma coherente de evitar que el futuro contamine el entrenamiento. [19]

### 12.3 Métricas obligatorias

```text
Performance:
- CAGR
- annualized return
- cumulative return
- hit rate

Risk:
- annualized volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- maximum drawdown
- CVaR 95%
- downside deviation

Portfolio:
- turnover
- average holding period
- average number of holdings
- concentration
- sector exposure
- beta
- factor exposure
- gross/net exposure

Execution:
- transaction costs
- slippage
- market impact
- rejected trades
- unfilled trades

Benchmark:
- S&P 500
- equal-weight universe
- sector benchmark
- factor benchmark
```

### 12.4 Anti-overfitting

Todo experimento debe registrar:

```yaml
experiment_id: exp_2026_001
strategy_version: v0.3.1
data_version: pit_us_equities_2026_10_01
universe_version: us_large_mid_liquid_v1
start_date: 2010-01-01
end_date: 2026-01-01
execution_assumption: next_open
commission_bps: 1
slippage_bps: 5
random_seed: 42
git_commit: abc123
```

Las evaluaciones realistas deben documentar explícitamente universo, datos point-in-time, timing de ejecución, costes, spread, slippage, turnover y artefactos de código. Una revisión reciente de sistemas de trading destaca que asumir ejecución idealizada al mismo cierre o ignorar fricciones puede inflar materialmente las conclusiones de una estrategia. [23]

---

## 13. Machine Learning y MLOps

### 13.1 Uso permitido de ML en fase inicial

ML puede utilizarse para:

- Rankear activos por retorno relativo futuro.
- Estimar probabilidad de rendimiento positivo.
- Detectar regímenes de mercado.
- Clasificar riesgo de drawdown.
- Detectar anomalías de datos.
- Analizar cambios en texto de earnings calls, filings o guidance.
- Estimar costes, liquidez o probabilidad de fill.

### 13.2 Uso no permitido inicialmente

No usar ML para:

- Ejecutar operaciones sin restricciones determinísticas.
- Predecir “el precio exacto” de una acción.
- Reemplazar la validación walk-forward.
- Eliminar controles de liquidez, costes y exposición.
- Hacer recomendaciones sin explicación de features y riesgo.

### 13.3 Model registry

Cada modelo debe tener:

```text
model_name
model_version
training_data_version
feature_schema_version
train_start_date
train_end_date
validation_dates
test_dates
metrics
feature_importance
hyperparameters
git_commit
approval_status
deployed_at
retired_at
```

### 13.4 Model monitoring

Monitorear:

- Data drift.
- Feature drift.
- Score distribution drift.
- Cambios de cobertura.
- Deterioro de performance fuera de muestra.
- Aumento de turnover.
- Exposición inesperada a sectores/factores.
- Diferencia entre retorno esperado y realizado.
- Incidentes de datos o provider outages.

---

## 14. APIs principales

### 14.1 Rankings

```http
GET /v1/rankings?as_of=2026-10-05&universe=us_equities&limit=50
```

Respuesta:

```json
{
  "as_of_date": "2026-10-05",
  "strategy_version": "v0.1.0",
  "data_version": "pit_us_equities_2026_10_05",
  "items": [
    {
      "security_id": "FIGI_OR_INTERNAL_ID",
      "ticker": "XYZ",
      "decision": "STRONG_LONG",
      "composite_score": 91.4,
      "percentile": 0.94,
      "factor_scores": {
        "value": 84.2,
        "growth": 72.4,
        "profitability": 93.1,
        "momentum": 88.5,
        "revisions": 95.7
      },
      "risk_flags": [],
      "explanation": [
        "Top 6% del universo elegible",
        "Revisiones de EPS positivas",
        "ROIC y margen FCF superiores al sector",
        "Momentum relativo favorable"
      ]
    }
  ]
}
```

### 14.2 Ficha de activo

```http
GET /v1/securities/{security_id}/analysis?as_of=2026-10-05
```

Debe incluir:

- Precio y performance.
- Score compuesto e historial.
- Desglose de factores.
- Comparación sectorial.
- Señales activas.
- Riesgos.
- Datos faltantes.
- Cambios respecto a la semana/mes anterior.
- Eventos próximos.
- Elegibilidad long/short.
- Explicación legible para usuario final.

### 14.3 Propuesta de rebalanceo

```http
POST /v1/portfolios/{portfolio_id}/rebalance/proposal
```

Respuesta:

```json
{
  "rebalance_id": "rb_2026_10_05_001",
  "status": "PENDING_APPROVAL",
  "current_nav": 100000,
  "estimated_turnover": 0.18,
  "estimated_cost_usd": 126.40,
  "orders": [
    {
      "ticker": "XYZ",
      "side": "BUY",
      "current_weight": 0.02,
      "target_weight": 0.05,
      "delta_weight": 0.03,
      "reason": [
        "Nuevo ingreso top decile",
        "Deterioro de una alternativa en el mismo sector"
      ]
    }
  ],
  "risk_before": {},
  "risk_after": {}
}
```

El endpoint de propuesta no debe enviar órdenes reales. La ejecución debe ser un flujo separado con aprobación humana.

---

## 15. Seguridad y controles

### 15.1 Secrets

Nunca guardar API keys, credenciales de broker o tokens en el repositorio.

```text
Development:
- .env local
- 1Password / Doppler / Infisical

Production:
- AWS Secrets Manager, GCP Secret Manager o Vault
- rotación de secretos
- control de acceso por rol
```

### 15.2 Roles

```text
viewer:
- puede ver rankings, dashboards y reportes

analyst:
- puede lanzar investigaciones y backtests

portfolio_manager:
- puede crear propuestas de rebalanceo

approver:
- puede aprobar propuestas para paper trading o ejecución

admin:
- puede gestionar usuarios, providers y configuración
```

### 15.3 Kill switch

Debe existir una forma de detener operaciones, pipelines y alertas de ejecución:

```text
GLOBAL_TRADING_ENABLED=false
PAPER_TRADING_ENABLED=true
LIVE_TRADING_ENABLED=false
```

### 15.4 Auditoría

Registrar todas las acciones sensibles:

```text
user_id
action
timestamp
request_id
portfolio_id
strategy_version
data_version
before_state
after_state
approval_id
reason
```

---

## 16. Infraestructura y despliegue

### 16.1 Local development

```text
Docker Compose:
- PostgreSQL
- Redis
- MinIO / S3-compatible storage
- FastAPI
- Worker
- Prefect/Dagster
- Grafana
- Prometheus
- Frontend Next.js
```

### 16.2 Producción inicial

```text
Cloud:
- AWS, GCP o Azure

Compute:
- containers en ECS/Fargate, Cloud Run o Kubernetes ligero

Database:
- PostgreSQL administrado
- TimescaleDB opcional
- S3/GCS para Parquet y artefactos

Queue:
- Redis administrado o RabbitMQ

Orchestration:
- Prefect Cloud / Prefect server o Dagster

Observability:
- Grafana Cloud o stack propio
- Sentry para errores
```

### 16.3 Escalabilidad

No empezar con Kubernetes si el equipo es pequeño.

Ruta recomendada:

```text
Fase 1:
Docker Compose + PostgreSQL + DuckDB + FastAPI + Streamlit/Next.js

Fase 2:
Managed Postgres + object storage + Prefect + Redis + workers

Fase 3:
Container orchestration + autoscaling + warehouse + feature store

Fase 4:
Servicios especializados de ejecución, streaming y baja latencia
```

---

## 17. Calidad de código

### 17.1 Reglas

```text
- Python type hints obligatorios.
- Pydantic para modelos de entrada/salida.
- Ruff para linting y formatting.
- Mypy para type checking.
- Pytest para pruebas unitarias e integración.
- Hypothesis para pruebas basadas en propiedades.
- Pre-commit obligatorio.
- SemVer para versiones de paquetes.
- Conventional Commits.
- ADRs para decisiones arquitectónicas.
```

### 17.2 Tests críticos

```text
Data:
- no duplicates en claves primarias
- no dates future-dated
- available_at <= as_of_date
- valid corporate action adjustments

Factors:
- percentiles entre 0 y 1
- scores entre 0 y 100
- sin leakage futuro
- estabilidad frente a valores nulos

Portfolio:
- pesos suman 1
- ningún peso excede límite
- límites sectoriales respetados
- turnover dentro de máximo
- costos no negativos

Backtest:
- no same-bar execution si la señal usa close
- benchmark calculado con mismo calendario
- cartera histórica reproducible
- mismas entradas producen mismos resultados
```

---

## 18. Roadmap

### Sprint 1 — Fundaciones

- Configurar monorepo.
- Docker Compose local.
- PostgreSQL, DuckDB, MinIO y Redis.
- Ingesta de precios diarios.
- Security master.
- Logging, configuración y tests base.
- Primer dashboard simple de precios y universo.

### Sprint 2 — Datos point-in-time

- Ingesta de fundamentales.
- Esquema `available_at`.
- Corporate actions.
- Validaciones de calidad.
- Almacenamiento Parquet particionado.
- Dataset gold de precios y fundamentales.

### Sprint 3 — Factores y ranking

- Value.
- Growth.
- Profitability.
- Momentum.
- Normalización por sector.
- Composite score.
- API de rankings.
- Ficha básica de activo.

### Sprint 4 — Cartera y rebalanceo

- Equal weight.
- Inverse volatility.
- Restricciones de concentración.
- Bandas de no-operación.
- Propuesta de rebalanceo.
- Dashboard de pesos y exposición.

### Sprint 5 — Backtesting serio

- Event-driven backtester.
- Timing de fill `t+1`.
- Slippage y costes.
- Benchmarks.
- Walk-forward testing.
- Reportes comparativos.

### Sprint 6 — Risk engine

- Volatilidad.
- Drawdown.
- VaR/CVaR.
- Beta.
- Riesgo sectorial.
- Alertas.
- Límites de riesgo.

### Sprint 7 — ML opcional

- Baseline de LightGBM/CatBoost para ranking.
- Registro de modelos.
- Feature importance.
- Drift monitoring.
- Comparación contra estrategia multifactorial explicable.

### Sprint 8 — Paper trading

- Broker sandbox.
- Órdenes simuladas.
- Ledger de operaciones.
- Reconciliación diaria.
- Aprobación humana obligatoria.
- Kill switch.

---

## 19. Criterios de éxito

El sistema no se evaluará solo por CAGR.

Debe cumplir:

```text
Investigación:
- todos los resultados reproducibles
- datos point-in-time documentados
- resultados fuera de muestra
- costes y slippage incorporados
- benchmark comparable

Producto:
- ranking diario disponible
- explicación por decisión
- trazabilidad por señal
- propuesta de rebalanceo auditable
- alertas de riesgo claras

Operación:
- pipelines con retries y alertas
- datos validados
- cobertura de tests alta en lógica de riesgo
- aprobación humana antes de operar
- kill switch funcional
```

---

## 20. Disclaimer

Esta plataforma es una herramienta de análisis, investigación y apoyo a decisiones. No constituye asesoría financiera, recomendación personalizada ni promesa de desempeño. Las estrategias cuantitativas pueden perder dinero, los backtests no garantizan resultados futuros y las posiciones short, el apalancamiento y los derivados implican riesgos adicionales.

Antes de ofrecer señales, gestionar capital de terceros o integrar ejecución real, revisar requisitos legales, tributarios y regulatorios aplicables en Chile y en cada mercado objetivo.
