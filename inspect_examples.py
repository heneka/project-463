import os

import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import get_movielens_dataset, prepare_data_splits

from models.lightgcn import LightGCNModel

from train import get_positive_matches, preview_embedding, recommend_movies_for_user, train

REPORT_PATH = os.path.join(os.path.dirname(__file__), 'inspection_report.txt')


def load_movie_metadata(dataset):

    movie_path = os.path.join(dataset.raw_dir, 'movies.dat')

    movie_lookup = {}

    with open(movie_path, 'r', encoding='latin-1') as movie_file:

        for line in movie_file:

            (movie_id_str, title, genres) = line.rstrip('\n').split('::')

            movie_lookup[int(movie_id_str) - 1] = {'title': title, 'genres': genres}

    return movie_lookup


def format_movie(movie_lookup, movie_index):

    movie = movie_lookup.get(movie_index, {'title': f'Movie {movie_index}', 'genres': 'Unknown'})

    return f"{movie['title']} [{movie['genres']}]"


def pick_sample_user(test_data, num_users):

    user_mask = test_data.edge_label_index[0] < num_users

    positive_users = test_data.edge_label_index[0][user_mask & (test_data.edge_label == 1)]

    if positive_users.numel() == 0:

        return 0

    return int(positive_users[0].item())


def main():

    dataset = get_movielens_dataset()

    (train_data, val_data, test_data, num_users, num_movies) = prepare_data_splits(dataset)

    num_nodes = num_users + num_movies

    movie_lookup = load_movie_metadata(dataset)

    model = LightGCNModel(num_nodes=num_nodes, embedding_dim=64, num_layers=3)

    (history, metrics, model) = train(model=model, train_data=train_data, val_data=val_data, test_data=test_data, num_nodes=num_nodes, num_users=num_users, num_movies=num_movies, epochs=5, lr=0.005, device='cuda' if torch.cuda.is_available() else 'cpu', return_model=True)

    sample_user = pick_sample_user(test_data, num_users)

    (user_embedding, recommendations) = recommend_movies_for_user(model=model, edge_index=train_data.edge_index, user_id=sample_user, num_users=num_users, num_movies=num_movies, top_k=10, train_edge_index=train_data.edge_index)

    positive_movies = get_positive_matches(test_data.edge_label_index, test_data.edge_label, sample_user, num_users)

    report_lines = []

    report_lines.append(f'Sample user node id: {sample_user}')

    report_lines.append(f'User embedding preview: {preview_embedding(user_embedding)}')

    report_lines.append('')

    report_lines.append('Held-out positive movies for this user:')

    for movie_index in positive_movies[:10]:

        report_lines.append(f'  - {movie_index}: {format_movie(movie_lookup, movie_index)}')

    report_lines.append('')

    report_lines.append('Top recommended movie nodes:')

    for item in recommendations:

        match_flag = 'MATCH' if item['movie_index'] in positive_movies else ''

        report_lines.append(f"  - movie_index={item['movie_index']:<5} movie_node_id={item['movie_node_id']:<5} score={item['score']:<8} {format_movie(movie_lookup, item['movie_index'])} {match_flag}")

    report_lines.append('')

    report_lines.append(f"Validation recall after the quick run: {(history['val_recall'][-1] if history['val_recall'] else 'n/a')}")

    report_lines.append(f'Quick test metrics: {metrics}')

    report_text = '\n'.join(report_lines)

    print(report_text)

    with open(REPORT_PATH, 'w', encoding='utf-8') as report_file:

        report_file.write(report_text + '\n')

    print(f'\nSaved readable output to: {REPORT_PATH}')

if __name__ == '__main__':

    main()
