from random import randint

class RecList:

	def __init__(self, size):
		self.size = size
		self.fo = 0
		self.diversity = 0
		self.relevance = 0
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
		return "Items: "+str(self.item_list)+"\nF.O: "+str(self.fo)

	def get_result(self):
		return self.item_list, self.score_list

	def clear(self):
		self.size = 0
		self.fo = 0
		self.diversity = 0
		self.relevance = 0
		self.item_list.clear()
		self.score_list.clear()

	def create_from_base_rec(self, base_rec_list, base_rec_score_list):
		for i in range(self.size):
			self.item_list.append(base_rec_list[i])
			self.score_list.append(base_rec_score_list[i])

	def clone(self, other_rec_list):
		self.clear()
		self.item_list = other_rec_list.item_list.copy()
		self.score_list = other_rec_list.score_list.copy()
		self.size = other_rec_list.size
		self.fo = other_rec_list.fo
		self.relevance = other_rec_list.relevance
		self.diversity = other_rec_list.diversity

	def create_neighbour(self, base_rec_list, base_rec_list_size, base_rec_score_list):
		neighbour = RecList(self.size)
		neighbour.clone(self)

		index_1 = randint(0, self.size)
		index_2 = randint(0, self.size)
		while index_1 == index_2:
			index_2 = randint(0, self.size)

		index_3 = randint(0, base_rec_list_size)
		while base_rec_list[index_3] in neighbour:
			index_3 = randint(0, base_rec_list_size)

		index_4 = randint(0, base_rec_list_size)
		while index_3 == index_4 or base_rec_list[index_4] in neighbour:
			index_4 = randint(0, base_rec_list_size)

		neighbour.item_list[index_1] = base_rec_list[index_3]
		neighbour.score_list[index_1] = base_rec_score_list[index_3]
		neighbour.item_list[index_2] = base_rec_list[index_4]
		neighbour.score_list[index_2] = base_rec_score_list[index_4]

		return neighbour
	
	def add_item(self, item):
		if len(self.item_list)+1 == self.size:
			self.item_list.append(item)
			return 0
		else:
			return -1
	
	def remove_item(self, item):
		if item in self.item_list:
			self.item_list.remove(item)

	