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

def loadTennisData():

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

    # List of match dicts: loser -> 0, winner -> log(loserGames/loserGames)
    matches = []
    for line in lines:
        match = {}

        winnerGames = float(line["winner_games_won"])
        loserGames = float(line["loser_games_won"])
        totalGames = winnerGames + loserGames
        if totalGames == 0:
            continue
        # Fraction of games won minus 0.5
        winMargin = winnerGames / (winnerGames + loserGames) - 0.5
        # winMargin = 1 # Binary 

        loserName = " ".join(line["loser_name"].split())
        winnerName = " ".join(line["winner_name"].split())

        match[loserName] = 0.0
        match[winnerName] = winMargin
        matches.append(match)

    return matches
