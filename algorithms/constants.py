class geocat_constants:
	_NEIGHBOR_DISTANCE = 0.5#km
	_N = 80# temp list size
	_K = 20# final list size
	_VERY_SMALL_VALUE = -100 # used for objective function
	_DIV_GEO_CAT_WEIGHT = 0.5 # beta,this is here because of the work to be done on parameter customization for each user
	_DIV_WEIGHT=0.75 # lambda, geo vs cat_DIV_WEIGHT = 0.75 # lambda, geo vs cat

	@classmethod
	def get_neighbor_distance(self):
		return self._NEIGHBOR_DISTANCE
	
	@classmethod
	def get_n(self):
		return self._N

	@classmethod
	def get_k(self):
		return self._K

	@classmethod
	def get_very_small_value(self):
		return self._VERY_SMALL_VALUE

	@classmethod
	def get_div_geo_cat_weight(self):
		return self._DIV_GEO_CAT_WEIGHT

	@classmethod
	def get_div_weight(self):
		return self._DIV_WEIGHT
