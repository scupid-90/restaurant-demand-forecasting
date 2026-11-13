# Restaurant Sales Analysis: Customer Demand Prediction

Group 3 project for our Data Science class. We're using restaurant sales data to
predict demand for each menu item, so kitchens can prep closer to what they'll
actually sell. Less food thrown out, fewer sell-outs.

## Status

All project stages are done and in this repo.

- [x] Project proposal (`docs/week1_proposal.pptx`)
- [x] Dataset (`data/restaurant_sales_data.csv`)
- [x] Data cleaning and exploration (`notebooks/restaurant_sales_notebook.ipynb`)
- [x] Demand model and prep recommendations (`src/demand_model.py`)
- [x] Comparison against standard approaches (`src/week3_benchmark.py`)
- [x] Final results and forward-in-time test (`src/week8_final_analysis.py`)
- [x] Prep-planning dashboard (`dashboard/dashboard.html`)
- [x] Weekly presentations (`docs/`)

## Results

A random forest forecasts units sold per dish, then a cost-aware buffer turns the
forecast into a prep amount. Scored on 2,000 held-out records, against prepping
each dish's average sales ("par level"):

| Metric | Par level | Ours |
|---|---|---|
| Average miss per record (units) | 114 | 83 |
| Demand filled | 80.4% | 93.5% |
| Waste at equal service level (units) | 314k | 170k |
| Total cost (waste + lost margin), par = 100 | 100 | 63 |

On a forward-in-time test (train Jan to Oct 19, test Oct 20 to Jan 1) the forecast
error is again 27% smaller than par level. Gradient boosting scored slightly better
than our random forest on forecast error (81.5 vs 83.0). Limits: one year of public
data, sales rather than true demand, and no test in a real kitchen. Numbers come from
`results/*.json`.

## Front end

`dashboard/dashboard.html` is the front end: a single self-contained page (Sales &
Prep Planner). Open it in a browser, or view it live once GitHub Pages is on
(Settings > Pages > Deploy from branch `main`, folder `/ (root)`):
https://scupid-90.github.io/restaurant-demand-forecasting/

## Data

Restaurant sales dataset from Kaggle: 10,000 rows, 13 columns, 50 restaurants,
14 menu items, January 2024 to January 2025.

## Setup

```bash
pip install -r requirements.txt
python src/week8_final_analysis.py   # reproduces the final results
```

## Team

Group 3, Data Science class.
