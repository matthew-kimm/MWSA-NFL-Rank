import numpy as np
import pandas as pd
from functools import reduce


def get_winner_loser_score_diff(data: pd.DataFrame):
    """
    Convert HOME, AWAY, HOME_SCORE, AWAY_SCORE to WINNER, LOSER, SCORE_DIFF
    :param data:
    :return:
    """
    df = data.copy()
    df['SCORE_DIFF'] = df['HOME_SCORE'] - df['AWAY_SCORE']
    df['WINNER_SCORE'] = np.maximum(df['HOME_SCORE'], df['AWAY_SCORE'])
    df['LOSER_SCORE'] = np.minimum(df['HOME_SCORE'], df['AWAY_SCORE'])
    df['WINNER'] = np.where(df['SCORE_DIFF'] > 0, df['HOME'], df['AWAY'])
    df['LOSER'] = np.where(df['SCORE_DIFF'] < 0, df['HOME'], df['AWAY'])
    df['SCORE_DIFF'] = np.abs(df['SCORE_DIFF'])
    return df


def assign_index(teams: list):
    """
    Assign each team an index (sequentially) for easy matrix access
    :param teams:
    :return: maps team -> idx, idx -> team
    """
    label_to_idx = {team: idx for idx, team in enumerate(sorted(teams))}
    idx_to_label = {idx: team for team, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def get_year_data(data: pd.DataFrame, year: int):
    """
    Get train or test data for a particular year
    :param data:
    :param year: 2012-2017
    :return:
    """
    df = data.copy()
    df = df[df['YEAR'] == year]
    df = get_winner_loser_score_diff(df)
    teams = set(df['WINNER']).union(set(df['LOSER']))
    num_teams = len(teams)
    label_to_idx, idx_to_label = assign_index(teams)
    df['WINNER'] = df['WINNER'].map(label_to_idx)
    df['LOSER'] = df['LOSER'].map(label_to_idx)
    train = df[df['KIND'] == 'REG']
    test = df[df['KIND'] == 'POST']
    return train, test, idx_to_label, num_teams


def get_games_matrix_and_scores(data: pd.DataFrame, num_teams: int):
    """

    :param data:
    :param num_teams:
    :return:
    """
    num_games = data.shape[0]
    games_matrix = np.zeros((num_games, num_teams))
    games_matrix[np.arange(num_games), data['WINNER'].to_numpy()] = 1
    games_matrix[np.arange(num_games), data['LOSER'].to_numpy()] = -1
    scores = data['SCORE_DIFF']
    return games_matrix, scores


def get_win_loss_matrix(data: pd.DataFrame, num_teams: int):
    df = pd.pivot_table(data, values='SCORE_DIFF', index='LOSER', columns='WINNER', aggfunc='count')
    df = df.reindex(index=np.arange(num_teams), columns=np.arange(num_teams)).fillna(0).to_numpy()
    return df


def get_score_diff_matrix(data: pd.DataFrame, num_teams: int):
    df = pd.pivot_table(data, values='SCORE_DIFF', index='LOSER', columns='WINNER', aggfunc='sum')
    df = df.reindex(index=np.arange(num_teams), columns=np.arange(num_teams)).fillna(0).to_numpy()
    return df


def get_both_point_matrix(data: pd.DataFrame, num_teams: int):
    win_score = pd.pivot_table(data, values='WINNER_SCORE', index='LOSER', columns='WINNER', aggfunc='sum')
    win_score = win_score.reindex(index=np.arange(num_teams), columns=np.arange(num_teams)).fillna(0).to_numpy()

    win_count = pd.pivot_table(data, values='WINNER_SCORE', index='LOSER', columns='WINNER', aggfunc='count')
    win_count = win_count.reindex(index=np.arange(num_teams), columns=np.arange(num_teams)).fillna(0).to_numpy()

    lose_score = pd.pivot_table(data, values='LOSER_SCORE', index='WINNER', columns='LOSER', aggfunc='sum')
    lose_score = lose_score.reindex(index=np.arange(num_teams), columns=np.arange(num_teams)).fillna(0).to_numpy()

    lose_count = pd.pivot_table(data, values='LOSER_SCORE', index='WINNER', columns='LOSER', aggfunc='count')
    lose_count = lose_count.reindex(index=np.arange(num_teams), columns=np.arange(num_teams)).fillna(0).to_numpy()

    score_matrix = win_score + lose_score
    game_count = win_count + lose_count

    df = np.where(game_count > 0, score_matrix / game_count, 0)
    return df


def hits_iteration(matrix: np.array, xi: float = 0.1, tol: float = 10e-5, max_iter: int = 100) -> np.array:
    n = matrix.shape[0]
    matrix_xi = xi * matrix
    x = np.ones((n, 1))
    t = (x - xi) / n
    for i in range(max_iter):
        xnew = matrix_xi @ x + t
        xnew = xnew / np.linalg.norm(xnew, 1)
        if np.linalg.norm(xnew - x, 1) < tol:
            break
        x = xnew
    return xnew


def offense_defense_iteration(matrix: np.array, xi: float = 0.1, tol: float = 10e-5, max_iter: int = 100) -> np.array:
    n = matrix.shape[0]
    mat = matrix + xi
    x = np.ones((n, 1))
    for i in range(max_iter):
        xnew = mat @ np.divide(1, mat.T @ (np.divide(1, x)))
        xnew = xnew / np.linalg.norm(xnew, 1)
        if np.linalg.norm(xnew - x, 1) < tol:
            break
        x = xnew
    y = mat.T @ np.divide(1, x)
    return xnew, y


def markov_iteration(matrix: np.array, xi: float = 0.85, tol: float = 10e-5, max_iter: int = 100):
    n = matrix.shape[0]
    matrix_new = xi * matrix + (1 - xi)/n
    x = np.ones((1, n)) / n
    for i in range(max_iter):
        x_new = x@matrix_new
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new.T
        x = x_new
    return x_new.T


def method_random(data: np.array, num_teams: int):
    rating = np.random.rand(num_teams)
    return rating


def method_massey(data: np.array, num_teams: int):
    X, y = get_games_matrix_and_scores(data, num_teams)
    M = X.T@X
    M[-1, :] = 1
    p = X.T@y
    p[-1] = 0
    rating = np.linalg.solve(M, p)
    return rating


def method_colley(data: np.array, num_teams: int):
    X, _ = get_games_matrix_and_scores(data, num_teams)
    M = X.T@X
    C = 2 * np.identity(M.shape[0]) + M
    b = 1 + 0.5 * np.sum(X, axis=0).T
    rating = np.linalg.solve(C, b)
    return rating


def method_hits_win(data: np.array, num_teams: int):
    R = get_win_loss_matrix(data, num_teams)
    H = R.T@R
    rating = hits_iteration(H)
    return rating


def method_hits_win_weight(data: np.array, num_teams: int):
    R = get_win_loss_matrix(data, num_teams)
    win = np.sum(R, axis=0).flatten()
    loss = np.sum(R, axis=1).flatten()
    total = win + loss
    pct = win / total
    weight = np.repeat(pct.reshape((num_teams, 1)), num_teams, axis=1)
    wR = weight*R
    H = wR.T@wR
    rating = hits_iteration(H)
    return rating


def method_hits_score(data: np.array, num_teams: int):
    S = get_score_diff_matrix(data, num_teams)
    H = S.T@S
    rating = hits_iteration(H)
    return rating


def method_win_pct(data: np.array, num_teams: int):
    R = get_win_loss_matrix(data, num_teams)
    win = np.sum(R, axis=0).flatten()
    loss = np.sum(R, axis=1).flatten()
    total = win + loss
    pct = win / total
    rating = pct + np.random.rand(num_teams) / (num_teams + 1)
    return rating


def method_offense_defense(data: np.array, num_teams: int):
    P = get_both_point_matrix(data, num_teams)
    defense_rating, offense_rating = offense_defense_iteration(P)
    rating = offense_rating / defense_rating
    return rating


def method_offense_defense_score_diff(data: np.array, num_teams: int):
    S = get_score_diff_matrix(data, num_teams)
    defense_rating, offense_rating = offense_defense_iteration(S)
    rating = offense_rating / defense_rating
    return rating


def method_markov_win(data: np.array, num_teams: int):
    R = get_win_loss_matrix(data, num_teams)
    no_loss_teams = np.where(~R.any(axis=1))[0]
    R[no_loss_teams, :] = 1 / num_teams
    M = R / np.sum(R, axis=1, keepdims=True)
    ratings = markov_iteration(M)
    return ratings


def method_markov_win_weight(data: np.array, num_teams: int):
    R = get_win_loss_matrix(data, num_teams)
    win = np.sum(R, axis=0).flatten()
    loss = np.sum(R, axis=1).flatten()
    total = win + loss
    pct = win / total
    weight = np.repeat(pct.reshape((num_teams, 1)), num_teams, axis=1) + 0.01
    wR = weight*R
    # just a sample adjustment to show markov win can be adjusted
    # only counts wins against opponent with above a certain overall win percentage (may help rank some teams but likely many ties)
    wR = np.where(wR >= 0.6, wR, 0)
    no_loss_teams = np.where(~wR.any(axis=1))[0]
    wR[no_loss_teams, :] = 1 / num_teams
    M = wR / np.sum(wR, axis=1, keepdims=True)
    ratings = markov_iteration(M)
    # adjust for ties
    ratings = ratings.flatten() + np.random.rand(num_teams) / (num_teams + 1)
    return ratings


def method_markov_score(data: np.array, num_teams: int):
    S = get_score_diff_matrix(data, num_teams)
    no_loss_teams = np.where(~S.any(axis=1))[0]
    S[no_loss_teams, :] = 1 / num_teams
    M = S / np.sum(S, axis=1, keepdims=True)
    ratings = markov_iteration(M)
    return ratings


def method_hits_agg(data: np.array, num_teams: int):
    m = method_massey(data, num_teams)
    c = method_colley(data, num_teams)

    massey_agg_row = np.repeat(m.reshape((num_teams, 1)), num_teams, axis=1)
    massey_agg_col = massey_agg_row.T
    massey_agg = (massey_agg_row < massey_agg_col).astype(int)
    colley_agg_row = np.repeat(c.reshape((num_teams, 1)), num_teams, axis=1)
    colley_agg_col = colley_agg_row.T
    colley_agg = (colley_agg_row < colley_agg_col).astype(int)

    A = massey_agg + colley_agg
    H = A.T@A
    ratings = hits_iteration(H)
    return ratings


def method_cohits(data: np.array, num_teams: int):
    S = get_score_diff_matrix(data, num_teams)
    x0 = method_win_pct(data, num_teams)
    x0 = x0 / np.linalg.norm(x0, 1)
    y0 = x0

    yi = y0

    lu = 0.0
    lv = 0.85

    for i in range(100):
        x = (1 - lu) * x0 + lu * S @ yi
        y = (1 - lv) * y0 + lv * S.T @ x
        y = y / np.linalg.norm(y)
        if np.linalg.norm(y - yi, 1) < 10e-5:
            break
        yi = y
    rating = y
    return rating


def method_cohits_agg(data: np.array, num_teams: int):
    m = method_massey(data, num_teams)
    c = method_colley(data, num_teams)

    massey_agg_row = np.repeat(m.reshape((num_teams, 1)), num_teams, axis=1)
    massey_agg_col = massey_agg_row.T
    massey_agg = (massey_agg_row < massey_agg_col).astype(int)
    colley_agg_row = np.repeat(c.reshape((num_teams, 1)), num_teams, axis=1)
    colley_agg_col = colley_agg_row.T
    colley_agg = (colley_agg_row < colley_agg_col).astype(int)

    A = massey_agg + colley_agg
    H = A.T @ A
    x0 = method_offense_defense_score_diff(data, num_teams)
    x0 = x0 / np.linalg.norm(x0, 1)
    y0 = x0

    yi = y0

    lu = 0.0
    lv = 0.85

    for i in range(100):
        x = (1 - lu) * x0 + lu * H @ yi
        y = (1 - lv) * y0 + lv * H.T @ x
        y = y / np.linalg.norm(y)
        if np.linalg.norm(y - yi, 1) < 10e-5:
            break
        yi = y
    rating = y
    return rating


def accuracy(data: pd.DataFrame, method: str, rating: np.array) -> pd.DataFrame:
    """

    :param data: test data
    :param method:
    :param rating:
    :return:
    """
    df = data.copy()
    rating_map = {idx: rate for idx, rate in enumerate(rating)}
    rank_map = pd.Series(rating_map).rank(ascending=False)
    num_games = df.shape[0]
    df[f'{method}_WINNER_RATING'] = df['WINNER'].map(rating_map)
    df[f'{method}_WINNER_RANK'] = df['WINNER'].map(rank_map)
    df[f'{method}_LOSER_RATING'] = df['LOSER'].map(rating_map)
    df[f'{method}_LOSER_RANK'] = df['WINNER'].map(rank_map)
    df[f'{method}_CORRECT'] = (df['WINNER'].map(rating_map) > df['LOSER'].map(rating_map)).astype(int)
    df[f'{method}_COMPATIBLE'] = (df['WINNER'].map(rating_map) == df['LOSER'].map(rating_map)).astype(int)
    df[f'{method}_INCORRECT'] = (df['WINNER'].map(rating_map) < df['LOSER'].map(rating_map)).astype(int)
    correct = sum(df[f'{method}_CORRECT'])
    print(f'{method}: {correct} / {num_games} :: {round(100 * correct / num_games, 2)}')
    return df[[col for col in df.columns if col.startswith(f'{method}')]]


if __name__ == '__main__':
    np.random.seed(1)
    nfl_data = pd.read_csv('nfl_data.csv')
    years = sorted(list(nfl_data['YEAR'].unique()))

    methods = {
               'RANDOM': method_random,
               'MASSEY': method_massey,
               'COLLEY': method_colley,
               'HITS-WIN': method_hits_win,
               'HITS-SCORE': method_hits_score,
               'HITS-WIN-WEIGHT': method_hits_win_weight,
               'WIN-PCT': method_win_pct,
               'OFFENSE-DEFENSE': method_offense_defense,
               'OFFENSE-DEFENSE-SCORE-DIFF': method_offense_defense_score_diff,
               'MARKOV-WIN': method_markov_win,
               'MARKOV-SCORE': method_markov_score,
               'MARKOV-ADJUST': method_markov_win_weight,
               'HITS-AGG': method_hits_agg,
               'COHITS': method_cohits,
               'COHITS-AGG': method_cohits_agg
    }

    year_result_data = []
    year_result_rating_data = []
    for year in years:
        print(f'\n{year}\n_____\n')
        year_train_data, year_test_data, label_map, n_teams = get_year_data(nfl_data, year)
        year_method_data = [year_test_data]
        for method, method_func in methods.items():
            method_rating = method_func(year_train_data, n_teams)
            team_ratings = pd.DataFrame(method_rating, columns=[f'{year}_{method}'])
            team_ratings.index = team_ratings.index.map(label_map)
            year_result_rating_data.append(team_ratings)
            year_accuracy_data = accuracy(year_test_data, method, method_rating)
            year_method_data.append(year_accuracy_data)
        year_result_data.append(pd.concat(year_method_data, axis=1))

    result = pd.concat(year_result_data)
    rating_result = reduce(lambda x, y: x.join(y), year_result_rating_data)
    rating_result = rating_result.rank(ascending=False)
    result.to_csv('method_accuracy_result.csv', index=False)
    rating_result.to_csv('method_rank_result.csv', index=True)
    correct_cols = [col for col in result.columns if col.endswith('_CORRECT')]

    correct = result[correct_cols].sum() / result.shape[0]
    print('\n')
    print(correct.T)
