import csv
import random
import math


class COPRecommending:
    def __init__(self, file_path):
        self.films = self.read_data_from_csv(file_path)
        self.best_selection = []
        self.best_score = -1
        self.genre_threshold = 0.122
        self.popularity_threshold = 0.172
        self.vote_average_threshold = 0.72
        self.initial_temperature = 1000
        self.min_temperature = 0.001
        self.cooling_rate = 0.99

    def read_data_from_csv(self, file_path):
        films = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                film_data = {
                    "title": row['original_title'],
                    "genres": float(row['genres']),
                    "vote_average": float(row['vote_average']),
                    "popularity": float(row['popularity']),
                    "ratio_likeable": float(row['ratio_likeable'])
                }
                films.append(film_data)
        return films

    def calculate_score(self, selection):
        popularity_sum = sum(film['popularity'] for film in selection)
        vote_average_sum = sum(film['vote_average'] for film in selection)
        return popularity_sum + vote_average_sum

    def generate_neighbor(self, selection):
        neighbor_selection = selection.copy()
        film_to_replace = random.choice(neighbor_selection)
        new_film = random.choice([film for film in self.films if
                                  film['genres'] == self.genre_threshold
                                  and film['popularity'] < self.popularity_threshold
                                  and film['vote_average'] <= self.vote_average_threshold])

        neighbor_selection[neighbor_selection.index(film_to_replace)] = new_film

        return neighbor_selection

    def simulated_annealing(self):
        current_selection = random.sample(self.films, len(self.films))
        current_score = self.calculate_score(current_selection)
        temperature = self.initial_temperature
        k = 1  # Bolztmann constant (appro. 1)

        while temperature > self.min_temperature:
            neighbor_selection = self.generate_neighbor(current_selection)
            neighbor_score = self.calculate_score(neighbor_selection)
            delta_score = neighbor_score - current_score

            #gibbs/bolztmann distribution
            acceptance_probability = math.exp(-delta_score / (k * temperature))

            if delta_score > 0 or random.random() < acceptance_probability:
                current_selection = neighbor_selection
                current_score = neighbor_score

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_selection = current_selection

            temperature *= self.cooling_rate

        return self.best_selection

    def recommend_films(self, k):
        self.simulated_annealing()

        if len(self.best_selection) >= k:
            return self.best_selection[:k]
        else:
            return self.best_selection
