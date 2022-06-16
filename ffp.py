from heuristics import *
import random
import math
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires
class DummyHyperHeuristic(HyperHeuristic):

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  #   nbRules = The number of rules to be contained in this hyper-heuristic  
  def __init__(self, features, heuristics, nbRules, seed):
    super().__init__(features, heuristics)
    random.seed(seed)
    self.conditions = []
    self.actions = []
    for i in range(nbRules):
      self.conditions.append([0] * len(features))
      for j in range(len(features)):
        self.conditions[i][j] = random.random()
      self.actions.append(heuristics[random.randint(0, len(heuristics) - 1)])
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    minDistance = float("inf")
    index = -1
    state = []
    for i in range(len(self.features)):
      state.append(problem.getFeature(self.features[i]))
    print("\t" + str(state))
    for i in range(len(self.conditions)):
      distance = self.__distance(self.conditions[i], state)      
      if (distance < minDistance):
        minDistance = distance
        index = i
    heuristic = self.actions[index] 
    print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
    return heuristic

  # Returns the string representation of this dummy hyper-heuristic
  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
    for i in range(len(self.conditions)):      
      text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"      
    return text

  # Returns the Euclidian distance between two vectors
  def __distance(self, vectorA, vectorB):
    distance = 0
    for i in range(len(vectorA)):
      distance += (vectorA[i] - vectorB[i]) ** 2
    distance = math.sqrt(distance)
    return distance

# Training Stage
# =====================
train_heuristics = False
dbg = False
train_hh = True
if train_heuristics:
  file_list = []
  results_ldeg = []
  results_gdeg = []
  for file in glob.glob("instances/BBGRL/*.in"):
    problem = FFP(file)
    file_list.append(os.path.basename(file))
    results_ldeg.append(problem.solve("LDEG", 1, debug=False))
    problem = FFP(file)
    results_gdeg.append(problem.solve("GDEG", 1, debug=False))

  df = pd.DataFrame(data={'file_name': file_list,
                          'ldeg': results_ldeg,
                          'gdeg': results_gdeg,
                          'id': list(range(len(file_list)))})

  df.to_csv('results_heuristics.csv')
  fig=plt.figure()
  ax1=fig.add_subplot(1,2,1)
  ax2=fig.add_subplot(1,2,2)
  df.plot.scatter(x='id', y='ldeg', c='g', label='ldeg', ax=ax1, sharex=ax1, sharey=ax1, ylim=(-0.1, 1.1))
  df.plot.scatter(x='id', y='gdeg', c='b', ax=ax2, label='gdeg', sharex=ax1, sharey=True, ylim=(-0.1, 1.1))
  ax1.set_xlabel("Test Sample", fontsize='x-large')
  ax2.set_xlabel("Test Sample", fontsize='x-large')
  ax1.set_ylabel("Rate of burnt nodes", fontsize='x-large')
  plt.suptitle("Rate of burnt nodes for each heuristic", fontsize='xx-large')
  plt.savefig('comparison_ldeg_gdeg.png')

if dbg:
  # Solves the problem using heuristic LDEG and one firefighter
  fileName = "instances/BBGRL/50_ep0.1_0_gilbert_1.in"
  problem = FFP(fileName)
  #test = deepcopy(problem)
  print("LDEG = " + str(problem.solve("LDEG", 3, debug=True)))
  # print("LDEG = " + str(problem.solve_ga(["LDEG", "GDEG"], 1, debug=True)))

  # Solves the problem using heuristic GDEG and one firefighter
  problem = FFP(fileName)
  print("GDEG = " + str(problem.solve("GDEG", 3, debug=True)))

#problem = FFP(fileName)
#print("MDEG = " + str(problem.solve("MDEG", 3, debug=True)))

# Solves the problem using a randomly generated dummy hyper-heuristic
if train_hh:
  seed = random.randint(0, 1000)
  print(seed)
  for gens in [200, 100, 500]:
    cnt = 0
    file_list_hh = []
    best_hh = []
    score_hh = []
    for file in glob.glob("instances/BBGRL/*.in"):
      problem = FFP(file)
      file_list_hh.append(os.path.basename(file))
      hh2 = uGAHyperHeuristic(["EDGE_DENSITY", "BURNING_NODES", "NODES_IN_DANGER"], ["LDEG", "GDEG", "LDEG", "GDEG"],
                              problem, seed, max_gens=gens)
      hh, score = hh2.solve()
      score_hh.append(score)
      best_hh.append(hh)
      cnt += 1
      if cnt == 50:
        break

    df2 = pd.DataFrame(data={'file_name': file_list_hh,
                             'hh_score': score_hh,
                             'best_hh': best_hh,
                             'id': list(range(len(file_list_hh)))})
    df2.to_csv(f'results_hh_{gens}.csv')
