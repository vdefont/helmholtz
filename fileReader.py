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

# Returns dict with top players as keys, rank as index (starts at 0)
def getTopItems(fileName, numItems):
    file = open(fileName)
    reader = csv.reader(file, delimiter = ',')
    topItems = {}
    curI = 0
    for lineItems in reader:
        name = lineItems[0]
        nameFormatted = " ".join(name.split())
        topItems[nameFormatted] = curI
        curI += 1
        if len(topItems) >= numItems:
            break
    return topItems

# Returns list of dicts
# Variable names provided by header in first line of file
def readCsvFile(fileName):
    file = open(fileName, 'r')

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
    return lines

# weightMethod: UNIFORM, SETS, GAMES, TOURNEY
def loadTennisData(weightMethod = "UNIFORM", maxPlayers = 10):

    # Read lines from file

    lines = readCsvFile('data/tennis/2017_matches.csv')

    curPlayers = getTopItems('data/tennis/ranking', maxPlayers)

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

        # If both players aren't in our list, ignore the match
        if not (winnerName in curPlayers and loserName in curPlayers):
            continue

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

def loadGolfData(maxPlayers = 10):

    lines = readCsvFile('data/golf/golfData2018.csv')
    playerIds = getTopItems('data/golf/ranking', maxPlayers)

    # Dict: name -> (dict: player -> score)
    tournaments = {}
    for line in lines:
        tournament = line["tournament"]
        if tournament not in tournaments:
            tournaments[tournament] = {}

        name = line["player"]
        nameFormatted = " ".join(name.split())
        score = -1.0 * float(line["total"]) # Make negative: lower is better
        
        if nameFormatted in playerIds:
            tournaments[tournament][name] = score

    tournamentList = list(tournaments.values())
    weights = [1.0] * len(tournamentList) # Weight all tourneys equally

    return tournamentList, weights, playerIds

loadGolfData()
