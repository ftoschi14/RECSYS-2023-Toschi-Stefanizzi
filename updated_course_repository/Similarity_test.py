from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Similarity.Cython import Compute_Similarity_Cython
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import scipy.sparse as sps
import numpy as np
import pandas as pd

# row = np.array([0, 3, 1, 0])
# col = np.array([0, 3, 1, 2])
# data = np.array([4, 5, 7, 9])
# urm = sps.coo_matrix((data, (row, col)))

URM_path = "Data/data_train.csv"
URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                header=0,
                                dtype={0:int, 1:int, 2:int},
                                engine='python')

URM_all_dataframe.columns = ["user_id", "item_id", "interaction"]

def preprocess_data(ratings: pd.DataFrame):
    unique_users = ratings.user_id.unique()
    unique_items = ratings.item_id.unique()

    num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

    print(num_users, min_user_id, max_user_id)
    print(num_items, min_item_id, max_item_id)

    mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "user_id": unique_users})
    mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

    ratings = pd.merge(left=ratings,
                       right=mapping_user_id,
                       how="inner",
                       on="user_id")

    ratings = pd.merge(left=ratings,
                       right=mapping_item_id,
                       how="inner",
                       on="item_id")

    return ratings


#  Call preprocess data function
ratings = preprocess_data(URM_all_dataframe)

urm = sps.coo_matrix((ratings.interaction.values, (ratings.mapped_user_id.values, ratings.mapped_item_id.values)))

rec = SLIM_BPR_Cython(urm)
rec.fit()
#rec = ItemKNNCFRecommender()
#rec.fit()