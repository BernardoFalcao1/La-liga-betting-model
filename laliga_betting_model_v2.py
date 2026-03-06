# Goal differential as a new predictor for our data
model_df["goal_diff"]= model_df['FTHG']-model_df['FTAG']

print(model_df[["FTHG","FTAG", "goal_diff"]].head())

# create team-level dataframe

# Home teams
home_games = model_df[["Date","HomeTeam","goal_diff"]].copy()
home_games = home_games.rename(columns={"HomeTeam":"Team","goal_diff":"gd"})
home_games["gd"] = home_games["gd"]

# Away Teams
away_games = model_df[["Date","AwayTeam","goal_diff"]].copy()
away_games = away_games.rename(columns={"AwayTeam":"Team","goal_diff":"gd"})
away_games["gd"] = -away_games["gd"]

team_games = pd.concat([home_games, away_games]).sort_values("Date")

team_games["rolling_gd"] = team_games.groupby("Team")["gd"].shift(1).rolling(5).mean()

print(team_games.head(10))

# home team rolling strength
home_strength = team_games.rename(columns={
    "Team": "HomeTeam",
    "rolling_gd": "home_form"
})[["Date","HomeTeam","home_form"]]

# away team rolling strength
away_strength = team_games.rename(columns={
    "Team": "AwayTeam",
    "rolling_gd": "away_form"
})[["Date","AwayTeam","away_form"]]

model_df = model_df.merge(home_strength, on=["Date","HomeTeam"], how="left")
model_df = model_df.merge(away_strength, on=["Date","AwayTeam"], how="left")

print(model_df[["HomeTeam","AwayTeam","home_form","away_form"]].head(10))

df2 = model_df.dropna(subset=["home_form", "away_form"]).copy()

X2 = df2[["market_home_prob", "home_form", "away_form"]]
X2 = sm.add_constant(X2)

y2 = df2["home_win"]

model2 = sm.Logit(y2, X2).fit()
print(model2.summary())
print(df2.shape)

df2["model_prob"] = model2.predict(X2)

market_logloss = log_loss(y2, df2["market_home_prob"])
model_logloss = log_loss(y2, df2["model_prob"])

print("Market Log Loss:", market_logloss)
print("Model Log Loss:", model_logloss)

split_index = int(len(df2) * 0.8)

train = df2.iloc[:split_index].copy()
test = df2.iloc[split_index:].copy()

print("Train size:", len(train))
print("Test size:", len(test))

X_train2 = train[["market_home_prob","home_form","away_form"]]
X_train2 = sm.add_constant(X_train2)

y_train2 = train["home_win"]

model3 = sm.Logit(y_train2, X_train2).fit()

print(model3.summary())

X_test2 = test[["market_home_prob","home_form","away_form"]]
X_test2 = sm.add_constant(X_test2)

test["model_prob"] = model3.predict(X_test2)

print(test[["market_home_prob","model_prob"]].head())

# Computing the the test section against the previous computation
test["ev_home"] = test["model_prob"] * test["B365H"] - 1

bets = test[test["ev_home"] > 0].copy()

bets["profit"] = bets.apply(
    lambda row: row["B365H"] - 1 if row["home_win"] == 1 else -1,
    axis=1
)

print("Test bets:", len(bets))
print("Total profit:", bets["profit"].sum())
print("Average profit per bet:", bets["profit"].mean())
