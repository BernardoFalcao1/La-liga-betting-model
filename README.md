# Soccer Match Outcome Prediction and Betting Market Analysis

## Overview

This project analyzes soccer match outcomes in **La Liga** using historical match data and bookmaker odds to study how well betting markets predict match results. The goal is to evaluate whether incorporating **team performance metrics** improves probability predictions relative to bookmaker implied probabilities.

The project builds a probabilistic model that predicts the likelihood of a home team win using market odds and engineered features derived from recent team performance.

---

## Objectives

- Evaluate the predictive accuracy of bookmaker implied probabilities
- Engineer features representing **recent team performance**
- Build a probabilistic model to estimate match outcome probabilities
- Compare model predictions with market probabilities using **log-loss**
- Test betting strategies using **expected value calculations**
- Evaluate model performance using **out-of-sample backtesting**

---

## Dataset

Match data was obtained from [Football-Data.co.uk](https://www.football-data.co.uk/).

The dataset includes:

- Match date
- Home and away teams
- Goals scored
- Bookmaker odds (Bet365)

Example columns used in this project:

- `HomeTeam`
- `AwayTeam`
- `FTHG` (Full Time Home Goals)
- `FTAG` (Full Time Away Goals)
- `B365H`, `B365D`, `B365A` (Bet365 odds)


---

## Feature Engineering

Several predictive features were constructed from historical match results.

### Market Implied Probability

Bookmaker odds were converted into implied probabilities:

\[
P(HomeWin) = \frac{1}{Odds}
\]

These probabilities represent the market's belief about match outcomes.

---

### Team Form Metrics

Rolling averages over the previous **five matches** were calculated for each team:

- Goals scored (attack strength)
- Goals conceded (defensive strength)

These features capture recent team performance trends.

Example engineered features:

- `home_attack`
- `home_defense`
- `away_attack`
- `away_defense`
- `form_diff`

---

## Model

A **logistic regression model** was used to estimate the probability of a home win.

The model used the following predictors:

- Market implied probability
- Recent attacking form
- Recent defensive form
- Relative team form difference

The model estimates:

\[
P(HomeWin) = f(\text{market probability}, \text{team form features})
\]

---

## Evaluation Method

Model performance was evaluated using **log-loss**, a proper scoring rule for probabilistic predictions.

Lower log-loss indicates better calibrated probability predictions.

To avoid look-ahead bias, matches were split chronologically:

- **80% training data**
- **20% testing data**

This ensures that predictions are evaluated on **future matches the model has not seen**.

---

## Betting Strategy Evaluation

To test whether the model identifies potential market inefficiencies, expected value (EV) was calculated:

\[
EV = P_{model} \times Odds - 1
\]

Bets were placed only when the expected value was positive.

Performance was measured using:

- total profit
- average profit per bet
- out-of-sample results

---

## Key Findings

- Betting market probabilities are **highly efficient predictors of match outcomes**
- Team performance features slightly improve probability calibration
- Improvements in log-loss suggest the model captures additional information beyond the market
- However, profitable betting strategies remain difficult to achieve out-of-sample

---

## Tools and Libraries

Python libraries used:

- `pandas`
- `numpy`
- `statsmodels`
- `scikit-learn`
- `matplotlib`

---

## Future Improvements

Several extensions could further improve the model:

- Incorporating multiple seasons of data
- Using **Elo ratings** or team strength models
- Including **expected goals (xG)** metrics
- Modeling match score distributions with **Poisson regression**
- Monte Carlo simulations to analyze betting strategy risk

---

## Author

**Bernardo**
