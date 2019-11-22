import random
from Particle import Particle

class Swarm:

	def __init__(self, size):
		self.particles = []
		self.size = size

	def __getitem__(self, key):
		return self.particles[key]

	def add_particle(self, particle):
		if len(self.particles) < self.size:
			self.particles.append(particle)
			return 0
		else:
			return -1
	
	def create_particles(self, base_item_list, base_score_list, particle_size, base_rec_size):
		best_candidate = int(round(0.4 * self.size, 0))
		random_candidate = self.size - best_candidate

		# Best candidate
		count = 0
		for i in range(best_candidate):
			if (count + particle_size) < base_rec_size:
				particle = Particle(particle_size)
				for j in range(count, count + particle_size):
					particle.add_item(base_item_list[j], base_score_list[j]) < 0
				
				count = j
				particle.set_initial_best()
				self.add_particle(particle)
			
			else:
				random_candidate += best_candidate-i+1
				break

		# Random candidate
		for i in range(random_candidate):
			particle = Particle(particle_size)
			for j in range(particle_size):
				limit = len(base_item_list)-1
				position = random.randint(0, limit)
				while base_item_list[position] in particle:
					position = random.randint(0, limit)
				
				particle.add_item(base_item_list[position], base_score_list[position])
			
			particle.set_initial_best()
			self.add_particle(particle)
