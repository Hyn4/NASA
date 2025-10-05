# models/earth_similarity.py

import pandas as pd
import numpy as np

class EarthSimilarityCalculator:

    def __init__(self):
        self.earth_params = {
            'pl_bmasse': 1.0,
            'pl_rade':   1.0,
            'pl_orbper': 365.25,
            'st_teff':   5778,
            'st_rad':    1.0
        }
        self.weights = {
            'pl_bmasse': 0.57,
            'pl_rade':   0.57,
            'pl_orbper': 0.3,
            'st_teff':   5.58,
            'st_rad':    0.5
        }
        self.key_features = ['pl_bmasse', 'pl_rade', 'pl_orbper', 'st_teff', 'st_rad']

    def _calculate_single_esi(self, feature, value):
        x0 = self.earth_params[feature]
        w  = self.weights[feature]
        if value + x0 == 0:
            return 0.0
        return (1 - abs((value - x0)/(value + x0)))**w

    def _calculate_total_esi(self, row):
        esis = [self._calculate_single_esi(f, row[f]) for f in self.key_features]
        return np.prod(esis)

    def calculate_single_planet(self, pl_bmasse, pl_rade, pl_orbper, st_teff, st_rad):
        df = pd.DataFrame([{ 
            'pl_bmasse': pl_bmasse,
            'pl_rade':   pl_rade,
            'pl_orbper': pl_orbper,
            'st_teff':   st_teff,
            'st_rad':    st_rad
        }])
        df[self.key_features] = df[self.key_features].apply(pd.to_numeric, errors='coerce')
        return float(df.apply(self._calculate_total_esi, axis=1).iloc[0])
