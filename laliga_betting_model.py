# Puttig the data into a df
csv_url_25_26 = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"

df = pd.read_csv(csv_url_25_26)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nHead:\n", df.head())

# Clean DataFrame with only the necessary columns
model_df = df[[
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG", # full time home goals
    "FTAG", # full time away goals
    "B365H", # Bet365 odds for Home win
    "B365D", # Bet365 odds for Draw
    "B365A", # Bet365 odds for Away win

]].copy()

model_df["Date"] = pd.to_datetime(model_df["Date"], dayfirst=True)

model_df = model_df.sort_values("Date").reset_index(drop=True)

print(model_df.head())
print(model_df.shape)
print(model_df.isna().sum())

# Making a home win column indicator
model_df["home_win"] = (model_df["FTHG"] > model_df["FTAG"]).astype(int)

#Checking results
print(model_df.shape)
print(model_df[["FTHG", "FTAG", "home_win"]].head())
print("Home win rate:", model_df["home_win"].mean())

# Home odds to win
model_df["market_home_prob"] = 1/ model_df["B365H"]

print(model_df[["B365H", "market_home_prob"]].head())
print("Average market implied home prob:", model_df["market_home_prob"].mean())


X = model_df[["market_home_prob"]]
X = sm.add_constant(X)

y = model_df["home_win"]

model = sm.Logit(y, X).fit()

print(model.summary())

# Making a prediction column
model_df["model_prob"] = model.predict(X)

# Checking results
print(model_df[["market_home_prob", "model_prob"]].head())
print("Average model prob:", model_df["model_prob"].mean())

# Measuring the model accuracy compared to the market
market_logloss = log_loss(y, model_df["market_home_prob"])
model_logloss = log_loss(y, model_df["model_prob"])

print("Market Log Loss:", market_logloss)
print("Model Log Loss:", model_logloss)

# Test For profitability on bets through "expected value of betting on home teams"
model_df["ev_home"] = model_df["model_prob"] * model_df["B365H"] - 1

print(model_df[["model_prob","B365H","ev_home"]].head())
print("Positive EV bets:", (model_df["ev_home"] > 0).sum())

# Simulation of the bets

# Taking only the positive EV bets
bets = model_df[model_df["ev_home"] > 0].copy()

# Profit from each bet
bets["profit"] = bets.apply(
    lambda row: row["B365H"] - 1 if row["home_win"]==1 else -1,
    axis = 1
)

print("Number of bets:", len(bets))
print("Total profit", bets["profit"].sum())
print("Average profit per bet", bets["profit"].mean())

# Creating a testing split for the matches and training
split_index = int(len(model_df) * 0.8) # 80 will be used to train and 20 will be used for testing

train = model_df.iloc[:split_index].copy()
test = model_df.iloc[split_index:].copy()

print("Train size:", len(train))
print("Test size:", len(test))

# Training set results
X_train = train[["market_home_prob"]]
X_train = sm.add_constant(X_train)

y_train = train["home_win"]

model = sm.Logit(y_train, X_train).fit()

print(model.summary())

# Test set results
X_test = test[["market_home_prob"]]
X_test = sm.add_constant(X_test)

test["model_prob"] = model.predict(X_test)

print(test[["market_home_prob","model_prob"]].head())

#Computing expected val only on test set
test["ev_home"] = test["model_prob"] * test["B365H"] - 1

bets = test[test["ev_home"] > 0].copy()

bets["profit"] = bets.apply(
    lambda row: row["B365H"] - 1 if row["home_win"] == 1 else -1,
    axis=1
)

print("Test bets:", len(bets))
print("Total profit:", bets["profit"].sum())
print("Average profit per bet:", bets["profit"].mean())
