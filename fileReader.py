import csv
import math

# Returns a list of users. Each user is a dict of item ratings
def testData():
    u1 = {"A":3, "B":1}
    u2 = {"A":7, "C":6}
    u3 = {"A":5, "D":0}
    u4 = {"B":2, "C":1}
    u5 = {"B":9, "D":8}
    u6 = {"C":1, "D":3}
    users = [u1,u2,u3,u4,u5,u6]
    weights = [2,1,4,3,0,5]
    items = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5}
    return users, weights, items

def testData2():
    u1 = {"A":0,"B":1}
    u2 = {"B":0,"C":2}
    u3 = {"C":0,"A":3}
    u4 = {"B":0,"D":4}
    users = [u1,u2,u3,u4]
    weights = [1,2,3,4]
    items = {"A":0,"B":1,"C":2,"D":3}
    return users, weights, items

def testData3():
    u1 = {"A":0,"B":2}
    u2 = {"A":0,"C":3}
    u3 = {"B":2,"C":3}
    u4 = {"D":6,"A":0.5}
    u5 = {"D":6,"B":1.5}
    u6 = {"D":6,"C":3.5}
    users = [u1,u2,u3,u4,u5,u6]
    weights = None
    items = {"A":0,"B":1,"C":2,"D":3}
    return users, weights, items

def testData4():
    u1 = {"A":0,"B":1}
    u2 = {"A":0,"C":-3}
    u4 = {"B":0,"C":2}
    u5 = {"B":0,"D":4}
    u6 = {"C":0,"D":1}
    users = [u1,u2,u4,u5,u6]
    weights = [5,7,6,8,5]
    items = {"A":0,"B":1,"C":2,"D":3}
    return users, weights, items

# Same as test4, except with no zero edges 
def testData5():
    u1 = {"A":0,"B":1}
    u2 = {"A":0,"C":-3}
    u3 = {"A":0,"D":1}
    u4 = {"B":0,"C":2}
    u5 = {"B":0,"D":4}
    u6 = {"C":0,"D":1}
    users = [u1,u2,u3,u4,u5,u6]
    weights = [5,7,5,6,8,5]
    items = {"A":0,"B":1,"C":2,"D":3}
    return users, weights, items

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
