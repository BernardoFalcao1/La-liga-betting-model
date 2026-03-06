df2["form_diff"] = df2["home_form"] - df2["away_form"]

print(df2[["home_form","away_form","form_diff"]].head())

X3 = df2[["market_home_prob","form_diff"]]
X3 = sm.add_constant(X3)

y3 = df2["home_win"]

model4 = sm.Logit(y3, X3).fit()

print(model4.summary())

print(df2[["home_form","away_form","form_diff"]].head(10))

home_games = model_df[["Date","HomeTeam","FTHG","FTAG"]].copy()
home_games = home_games.rename(columns={"HomeTeam":"Team","FTHG":"scored","FTAG":"conceded"})

away_games = model_df[["Date","AwayTeam","FTAG","FTHG"]].copy()
away_games = away_games.rename(columns={"AwayTeam":"Team","FTAG":"scored","FTHG":"conceded"})

team_games = pd.concat([home_games, away_games]).sort_values("Date")

team_games["rolling_scored"] = team_games.groupby("Team")["scored"].shift(1).rolling(5).mean()
team_games["rolling_conceded"] = team_games.groupby("Team")["conceded"].shift(1).rolling(5).mean()

print(team_games.head(10))

home_stats = team_games.rename(columns={
    "Team": "HomeTeam",
    "rolling_scored": "home_attack",
    "rolling_conceded": "home_defense"
})[["Date","HomeTeam","home_attack","home_defense"]]

away_stats = team_games.rename(columns={
    "Team": "AwayTeam",
    "rolling_scored": "away_attack",
    "rolling_conceded": "away_defense"
})[["Date","AwayTeam","away_attack","away_defense"]]

df2 = df2.merge(home_stats, on=["Date","HomeTeam"], how="left")
df2 = df2.merge(away_stats, on=["Date","AwayTeam"], how="left")

print(df2[[
    "HomeTeam",
    "AwayTeam",
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense"
]].head())

df3 = df2.dropna(subset=[
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense"
]).copy()

print(df3.shape)
