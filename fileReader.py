import csv
import math

# Returns a list of users. Each user is a dict of item ratings
def testData():
    u1 = {"A":5, "B":4, "C":3}
    u2 = {"B":2}
    u3 = {"A":4, "B":3}
    u4 = {"B":4, "C":2}
    u5 = {"B":5, "C":2}
    users = [u1,u2,u3,u4,u5]
    return users

# weightMethod: UNIFORM, SETS, GAMES, TOURNEY
def loadTennisData(weightMethod = "UNIFORM", maxPlayers = 10):

    # Read lines from file

    file = open('2017_matches.csv', 'r')

    reader = csv.reader(file, delimiter=',')

    headers = None
    lines = []
    for lineItems in reader:
        if headers is None:
            headers = lineItems
        else:
            line = {}
            for i in range(len(headers)):
                line[headers[i]] = lineItems[i]
            lines.append(line)

    file.close()

    curPlayerI = 0
    curPlayers = {}

    # List of match dicts: loser -> 0, winner -> log(loserGames/loserGames)
    matches = []
    weights = []
    for line in lines:
        match = {}

        winnerGames = float(line["winner_games_won"])
        loserGames = float(line["loser_games_won"])
        totalGames = winnerGames + loserGames
        if totalGames == 0:
            continue

        winnerSets = float(line["winner_sets_won"])
        loserSets = float(line["loser_sets_won"])
        totalSets = winnerSets + loserSets

        tourneyRoundOrder = int(line["round_order"])
        tourneyWeight = 2 ** (1-tourneyRoundOrder)

        # Fraction of games won minus 0.5
        winMargin = winnerGames / (winnerGames + loserGames) - 0.5
        # winMargin = 1 # Binary

        loserName = " ".join(line["loser_name"].split())
        winnerName = " ".join(line["winner_name"].split())

        # If already reached max rows, stop
        if len(curPlayers) >= maxPlayers and not (loserName in curPlayers and winnerName in curPlayers):
            continue
        for name in [loserName, winnerName]:
            if name not in curPlayers:
                curPlayers[name] = curPlayerI
                curPlayerI += 1

        match[loserName] = 0.0
        match[winnerName] = winMargin
        matches.append(match)

        # Add weight
        if weightMethod == "UNIFORM":
            weights.append(1)
        elif weightMethod == "SETS":
            weights.append(totalSets)
        elif weightMethod == "GAMES":
            weights.append(totalGames)
        elif weightMethod == "TOURNEY":
            weights.append(tourneyWeight)

    return matches, weights, curPlayers
