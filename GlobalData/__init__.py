
import time
import chess
import pickle
import pandas as pd


class GlobalData:
    # data_columns = ['color', 'h1', 'h2', 'h3', 'h4', 'H']
    # input_columns = ['color', 'h1', 'h2', 'h3', 'h4']
    data_columns = ['h1', 'h2', 'h3', 'h4', 'H']
    input_columns = ['h1', 'h2', 'h3', 'h4']
    to_normalize_cols = ['h1', 'h2', 'h3', 'h4']
    results = []

    evaluated = dict()
    toplevel_evaluated = dict()

    heuristic_depth = 3
    show_display = True
    is_first_run = True

    predictor = None

    games_data = pd.DataFrame(columns=data_columns)
    # toplevel_eval = [dict(), dict()]
    # toplevel_eval[chess.WHITE] = dict()
    # toplevel_eval[chess.BLACK] = dict()

    @staticmethod
    def save_data():
        GlobalData.predictor.save_model()

        with open('data/depth'+str(GlobalData.heuristic_depth)+'/othervars.pkl', 'wb') as f:
            pickle.dump(
                [GlobalData.evaluated, GlobalData.toplevel_evaluated, GlobalData.games_data, GlobalData.results], f)

    @staticmethod
    def load_data(model=True):
        if model:
            GlobalData.predictor.load_model()
            print("Model loaded correctly")

        with open('data/depth'+str(GlobalData.heuristic_depth)+'/othervars.pkl', 'rb') as f:
            GlobalData.evaluated, GlobalData.toplevel_evaluated, GlobalData.games_data, GlobalData.results = pickle.load(
                f)
        print("Vars loaded correctly")
