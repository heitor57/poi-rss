from random import randint

class Particle:

	def __init__(self, size):
		self.size = size
		self.best_fo = 0
		self.best_relevance = 0
		self.best_diversity = 0
		self.best_item_list = []
		self.best_score_list = []

		self.item_list = []
		self.score_list = []

	def __eq__(self, other):
		return self.item_list == other.item_list

	def __ne__(self, other):
		return self.item_list != other.item_list

	def __len__(self):
		return self.size

	def __getitem__(self, key):
		return self.item_list[key]

	def __contains__(self, key):
		return key in self.item_list

	def __str__(self):
		return "Items: "+str(self.item_list)+"\nF.O: Unknown"

	def clear(self):
		self.best_fo = 0
		self.best_relevance = 0
		self.best_diversity = 0
		self.best_item_list = []
		self.best_score_list = []

		self.item_list = []
		self.score_list = []

	def clone(self, other_particle):
		self.clear()
		self.size = other_particle.size
		self.best_fo = other_particle.best_fo
		self.best_relevance = other_particle.best_relevance
		self.best_diversity = other_particle.best_diversity
		self.best_item_list = other_particle.best_item_list.copy()
		self.best_score_list = other_particle.best_score_list.copy()

		self.item_list = other_particle.item_list.copy()
		self.score_list = other_particle.score_list.copy()

	def clone_new_current(self, other_particle):
		self.item_list = other_particle.item_list.copy()
		self.score_list = other_particle.score_list.copy()

	def add_item(self, item, item_score):
		if len(self.item_list) < self.size:
			self.item_list.append(item)
			self.score_list.append(item_score)
			return 0
		else:
			return -1

	def set_initial_best(self):
		self.best_item_list = self.item_list.copy()
		self.best_score_list = self.score_list.copy()
