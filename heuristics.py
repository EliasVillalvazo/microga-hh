"""
Python file to store the class HyperHeuristic and the GA algorithm approach
"""
import numpy as np
import random

from copy import deepcopy

# Provides the methods to create and solve the firefighter problem
class FFP:

  # Constructor
  #   fileName = The name of the file that contains the FFP instance
  def __init__(self, fileName):
    file = open(fileName, "r")
    text = file.read()
    tokens = text.split()
    seed = int(tokens.pop(0))
    self.n = int(tokens.pop(0))
    model = int(tokens.pop(0))
    int(tokens.pop(0)) # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    self.state = [0] * self.n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
      b = int(tokens.pop(0))
      self.state[b] = -1
    self.graph = []
    for i in range(self.n):
      self.graph.append([0] * self.n)
    while tokens:
      x = int(tokens.pop(0))
      y = int(tokens.pop(0))
      self.graph[x][y] = 1
      self.graph[y][x] = 1

  # Solves the FFP by using a given method and a number of firefighters
  #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
  #   nbFighters = The number of available firefighters per turn
  #   debug = A flag to indicate if debugging messages are shown or not
  def solve(self, method, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))
    t = 0
    while (spreading):
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):
        heuristic = method
        if (isinstance(method, HyperHeuristic)):
          heuristic = method.nextHeuristic(self)
        node = self.__nextNode(heuristic)
        if (node >= 0):
          # The node is protected
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(t) + ": A firefighter protects node " + str(node))
      # It spreads the fire among the unprotected nodes
      spreading = False
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1):
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(t) + ": Fire spreads to node " + str(j))
      t = t + 1
      if (debug):
        print("---------------")
    if (debug):
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")


  def solve_ga(self, method, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))
    t = 0
    while (spreading):
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):
        heuristic = method[t] if len(method) >= t+1 else method[-1]
        if (isinstance(method, HyperHeuristic)):
          heuristic = method.nextHeuristic(self)
        node = self.__nextNode(heuristic)
        if (node >= 0):
          # The node is protected
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(t) + ": A firefighter protects node " + str(node))
      # It spreads the fire among the unprotected nodes
      spreading = False
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1):
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(t) + ": Fire spreads to node " + str(j))
      t = t + 1
      if (debug):
        print("---------------")
    if (debug):
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    #print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")
    #return self.getFeature("BURNING_EDGES")

  # Selects the next node to protect by a firefighter
  #   heuristic = A string with the name of one available heuristic
  def __nextNode(self, heuristic):
    index  = -1
    best = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        index = i
        break
    value = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        if (heuristic == "LDEG"):
          # It prefers the node with the largest degree, but it only considers
          # the nodes directly connected to a node on fire
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and self.state[j] == -1):
              value = sum(self.graph[i])
              break
        elif (heuristic == "GDEG"):
            value = sum(self.graph[i])
        elif heuristic == "MDEG":
            for j in range(len(self.graph[i])):
                if (self.graph[i][j] == 1 and self.state[j] == -1):
                    value = len(self.graph[i]) - sum(self.graph[i])
                    break

        else:
          print("=====================")
          print("Critical error at FFP.__nextNode.")
          print("Heuristic " + heuristic + " is not recognized by the system.")
          print("The system will halt.")
          print("=====================")
          exit(0)
      if (value > best):
        best = value
        index = i
    return index

  # Returns the value of the feature provided as argument
  #   feature = A string with the name of one available feature
  def getFeature(self, feature):
    f = 0
    if (feature == "EDGE_DENSITY"):
      n = len(self.graph)
      for i in range(len(self.graph)):
        f = f + sum(self.graph[i])
      f = f / (n * (n - 1))
    elif (feature == "AVG_DEGREE"):
      n = len(self.graph)
      count = 0
      for i in range(len(self.state)):
        if (self.state[i] == 0):
          f += sum(self.graph[i])
          count += 1
      if (count > 0):
        f /= count
        f /= (n - 1)
      else:
        f = 0
    elif (feature == "BURNING_NODES"):
      for i in range(len(self.state)):
        if (self.state[i] == -1):
          f += 1
      f = f / len(self.state)
    elif (feature == "BURNING_EDGES"):
      n = len(self.graph)
      for i in range(len(self.graph)):
        for j in range(len(self.graph[i])):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
      f = f / (n * (n - 1))
    elif  (feature == "NODES_IN_DANGER"):
      for j in range(len(self.state)):
        for i in range(len(self.state)):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
            break
      f /= len(self.state)
    else:
      print("=====================")
      print("Critical error at FFP._getFeature.")
      print("Feature " + feature + " is not recognized by the system.")
      print("The system will halt.")
      print("=====================")
      exit(0)
    return f

  # Returns the string representation of this problem
  def __str__(self):
    text = "n = " + str(self.n) + "\n"
    text += "state = " + str(self.state) + "\n"
    for i in range(self.n):
      for j in range(self.n):
        if (self.graph[i][j] == 1 and i < j):
          text += "\t" + str(i) + " - " + str(j) + "\n"
    return text



# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation
class HyperHeuristic:
    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    def __init__(self, features, heuristics):
        if (features):
            self.features = features.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of features cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)
        if (heuristics):
            self.heuristics = heuristics.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of heuristics cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        print("=====================")
        print("Critical error at HyperHeuristic.nextHeuristic.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)

    # Returns the string representation of this hyper-heuristic
    def __str__(self):
        print("=====================")
        print("Critical error at HyperHeuristic.__str__.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)


class uGAHyperHeuristic(HyperHeuristic):

    def __init__(self, features, heuristics, problem, seed, population=5, individual_size=6, max_gens=10):
        super().__init__(features, heuristics)
        np.random.seed(seed)
        self.ind_length = individual_size
        self.max_gens = max_gens
        self.popu_size = population
        self.problem = problem

        self.debug = False
        self.ff = 1
        self.goal = 0.90

        # Create the first generation of individuals to test
        self.population = self.create_population()
        self.evaluation = [None] * self.popu_size

    def __str__(self):
        return self.number_to_string(self.population[0])

    def create_individual(self):
        """
        Creates an individual of size self.size where each position corresponds to a
        certain heuristic
        """
        return np.random.randint(low=0, high=len(self.heuristics), size=self.ind_length).tolist()

    def create_population(self):
        """
        Creates a population of self.popu_size individuals
        :return: population
        """
        population : list = [None] * self.popu_size
        for i in range(self.popu_size):
            population[i] = self.create_individual()

        return population

    def combine(self, a, b, c_rate=1.0):
        """
        Combine two individuals and generate two offspring. Combine is 1-point crossover
        :param a: first individual (list)
        :param b: second individual (list)
        :param c_rate: probability to combine individuals
        :return: two lists of combined offspring
        """
        if random.random() < c_rate:
            cross_point = np.random.randint(1, self.ind_length)
            offspring_a = np.append(a[0:cross_point], b[cross_point:])
            offspring_b = np.append(b[0:cross_point], a[cross_point:])
        else:
            offspring_a = np.copy(a)
            offspring_b = np.copy(b)

        return offspring_a.tolist(), offspring_b.tolist()


    def number_to_string(self, individual):
        """
        Transform an individual to the heuristic representation
        :return:
        """
        out = [None] * self.ind_length
        for i, element in enumerate(individual):
            out[i] = self.heuristics[element]
        return out


    def evaluate(self, individual, problem_instance : FFP):
        """
        Evaluates the firefighter problem for the proposed hyper-heuristic individual
        :param individual: proposed hyper-heuristic
        :param problem_instance: instance of the problem that wants to be solved
        :return: given the solve_ga method. By default it returns only the burning nodes percent
        """
        if self.debug:
            print(f"testing {individual}")
        return problem_instance.solve_ga(individual, self.ff, self.debug)

    def tournament(self):
        rivals = random.sample(range(self.popu_size), 2)
        if self.evaluation[rivals[1]] < self.evaluation[rivals[0]]:
            return rivals[1]
        else:
            return rivals[0]

    def create_couples(self, winners):
        couple_1, couple_2 = [], []
        tmp = deepcopy(winners)
        # fill the first couple
        couple_1.append(tmp.pop(0))
        for i in range(0, len(tmp)):
            if couple_1[0] != winners[i]:
                couple_1.append(tmp.pop(i))
                break

        if len(couple_1) == 1:
            couple_1.append(couple_1[0] - 1)

        if len(set(tmp)) == 1:
            tmp[-1] = self.popu_size - 1 if tmp[-1] != self.popu_size - 1 else self.popu_size - 2

        return couple_1, tmp


    def solve(self):
        """
        Solver function that returns the best hyper heuristic combination from the evolutionary process
        :return:
        """
        best_yet = float("inf")
        index = 10
        new_population = [None] * self.popu_size
        # Initial Population is already created, we just need to evaluate each individual
        for i, individual in enumerate(self.population):
            self.evaluation[i] = self.evaluate(self.number_to_string(individual), deepcopy(self.problem))
            if self.evaluation[i] < best_yet:
                index = i
                best_yet = self.evaluation[i]
                # the best individual will be copied directly to the next generation
                elite = deepcopy(individual)


        create_new = 1
        for t in range(self.max_gens):
            print(t)
            # start evolutionary process. Let individuals compete in random tournaments
            # only four elements will be selected
            if create_new % 5 == 0:
                # create a whole new population
                new_population = self.create_population()
                new_population[-1] = elite

            else:
                # otherwise combine the current set
                winners = []
                for i in range(4):
                    winners.append(self.tournament())

                new_population[0] = elite
                couple_a, couple_b = self.create_couples(winners)
                new_population[1], new_population[2] = self.combine(self.population[couple_a[0]], self.population[couple_a[1]])
                new_population[3], new_population[4] = self.combine(self.population[couple_b[0]], self.population[couple_b[1]])

            self.population = deepcopy(new_population)
            for i, individual in enumerate(self.population):
                self.evaluation[i] = self.evaluate(self.number_to_string(individual), deepcopy(self.problem))
                if self.evaluation[i] < best_yet:
                    index = i
                    best_yet = self.evaluation[i]
                    # the best individual will be copied directly to the next generation
                    elite = deepcopy(individual)

            if best_yet < self.goal:
                break

            create_new += 1
        print(f"Best Hyper Heuristic found: {elite}")
        print(f"Evaluation is {self.evaluate(self.number_to_string(elite), deepcopy(self.problem))}")
        return self.number_to_string(elite), self.evaluate(self.number_to_string(elite), deepcopy(self.problem))