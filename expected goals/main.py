# ===============================
# Probabilistic Expected Points Model (NFL Plays)
# ===============================


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Dani\Downloads\nfl_data_testing_skills.csv")


df['TD'] = df['actionType'].str.contains('touchdown', case=False).astype(int)
df['FG'] = df['actionType'].str.contains('field goal', case=False).astype(int)
df['Safety'] = df['actionType'].str.contains('safety', case=False).astype(int)
df['Turnover'] = df['actionType'].str.contains('interception|fumble', case=False).astype(int)
df['Punt'] = df['actionType'].str.contains('punt', case=False).astype(int)


features = [
    "down", "distance", "yardLine", "yardsToEndzone",
    "quarter", "halves", "homeTeamScore", "awayTeamScore"
]
X = df[features]

classifiers = {}
events = {'TD':7, 'FG':3, 'Safety':-2, 'Turnover':0, 'Punt':0}

for event in events.keys():
    y = df[event]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    classifiers[event] = clf
    print(f"{event} classifier trained. Test accuracy: {clf.score(X_test, y_test):.3f}")

def compute_ep_probabilistic(play_before, play_after=None, classifiers=classifiers, events=events):
    """
    Compute Expected Points (EP) probabilistically using multiple classifiers.
    """
    def ep_for_play(play):
        play_df = pd.DataFrame([play])[features]
        ep = 0
        for event, points in events.items():
            prob = classifiers[event].predict_proba(play_df)[0][1]  # probability of event
            ep += prob * points
        return ep

    ep_before = ep_for_play(play_before)
    result = {"EP_before": ep_before}

    if play_after:
        ep_after = ep_for_play(play_after)
        result["EP_after"] = ep_after
        result["Delta_EP"] = ep_after - ep_before

    return result


play_before = {
    "down": 2,
    "distance": 5,
    "yardLine": 35,
    "yardsToEndzone": 65,
    "quarter": 2,
    "halves": 1,
    "homeTeamScore": 14,
    "awayTeamScore": 10
}

play_after = {
    "down": 1,
    "distance": 10,
    "yardLine": 25,
    "yardsToEndzone": 10,
    "quarter": 2,
    "halves": 1,
    "homeTeamScore": 14,
    "awayTeamScore": 10
}

result = compute_ep_probabilistic(play_before, play_after)
print(result)

