"""

See https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/
See https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares

"""

import numpy as np
import pandas as pd

MIN_POSSIBLE_RATING = 1
MAX_POSSIBLE_RATING = 5


class Recommender(object):
    def __init__(self, ratings_df, num_factors):
        self.ratings_df = ratings_df
        self.user_item_matrix = ratings_df \
            .pivot_table(index='user_id',
                         columns='item_id',
                         values='rating') \
            .fillna(0) \
            .values
        self.num_users, self.num_items = self.user_item_matrix.shape
        self.num_factors = num_factors
        self.user_matrix = np.random.rand(self.num_users, num_factors)
        self.item_matrix = np.random.rand(self.num_items, num_factors)
        self.est_user_item_matrix = self.predict_ratings_matrix()
        self.recommendations = self.compute_recommendations()

    def compute_loss(self):
        error = (self.user_item_matrix - self.predict_ratings_matrix()) ** 2
        return np.sum(error ** 2)

    def predict_ratings_matrix(self):
        return np.matmul(self.user_matrix, self.item_matrix.transpose())

    def train_user_item_matrix(self, num_iter):
        for i in range(num_iter):
            self.user_matrix = np.linalg.solve(
                np.matmul(self.item_matrix.transpose(),
                          self.item_matrix),
                np.matmul(self.item_matrix.transpose(),
                          self.user_item_matrix.transpose())
            ).transpose()

            self.item_matrix = np.linalg.solve(
                np.matmul(self.user_matrix.transpose(),
                          self.user_matrix),
                np.matmul(self.user_matrix.transpose(),
                          self.user_item_matrix)
            ).transpose()

        self.est_user_item_matrix = self.predict_ratings_matrix()

    @staticmethod
    def normalize_ratings(est_user_item_matrix):
        """Numbers seem a little weird"""
        min_rating = np.min(est_user_item_matrix)
        max_rating = np.max(est_user_item_matrix)
        return (est_user_item_matrix - min_rating) / \
               (max_rating - min_rating) * \
               (MAX_POSSIBLE_RATING - MIN_POSSIBLE_RATING) + \
               MIN_POSSIBLE_RATING

    def compute_recommendations(self):
        recommendations = np.argmax(self.est_user_item_matrix, axis=1)
        {user_id: item_id for user_id, item_id in zip(range(self.num_users), recommendations)}
