from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial.distance import euclidean

app = Flask(__name__)

# Funzione per ottenere i film in base agli attributi di input
def get_movies(df, attributes):
    mask = pd.Series([False] * len(df), index=df.index)
    for column, value in attributes.items():
        if column in ['genres', 'primaryName', 'actor_names']:
            if value:
                mask |= df[column].str.split(',').apply(lambda x: bool(set(x) & set(value)))
        elif column == 'startYear':
            df[column] = df[column].replace('\\N', np.nan)
            df[column] = pd.to_numeric(df[column], errors='coerce')
            start, end = value
            temp_mask = pd.Series([True] * len(df))
            if start:
                temp_mask &= df[column] >= float(start)
            if end:
                temp_mask &= df[column] <= float(end)
            mask |= temp_mask
    return df[mask]

# Funzione per ottenere i primi 20 film in base al rating e al numero di voti
def get_top_movies(df, n=20):
    df = df.dropna(subset=['averageRating', 'numVotes'])
    df = df.sort_values(['averageRating', 'numVotes'], ascending=[False, False])
    return df.head(n)

# Funzione per calcolare i pesi
def calculate_weights(df):
    df = df.dropna(subset=['actors', 'directors', 'genres', 'startYear'])
    total_movies = df['tconst'].nunique()

    actor_counts = pd.Series([actor for sublist in df['actors'].str.split(',') for actor in sublist]).value_counts()
    genre_counts = pd.Series([genre for sublist in df['genres'].str.split(',') for genre in sublist]).value_counts()

    Wa = actor_counts / total_movies
    Wd = df['directors'].value_counts() / total_movies
    Wg = genre_counts / total_movies
    Wy = df['startYear'].value_counts() / total_movies

    def calculate_wr(row):
        rating = row['averageRating']
        votes = row['numVotes']
        if rating >= 10:
            return 10 if votes <= 1000 else 20 if votes <= 10000 else 30
        elif rating >= 9:
            return 9 if votes <= 1000 else 18 if votes <= 10000 else 27
        elif rating >= 8:
            return 8 if votes <= 1000 else 16 if votes <= 10000 else 24
        elif rating >= 7:
            return 7 if votes <= 1000 else 14 if votes <= 10000 else 21
        elif rating >= 6:
            return 6 if votes <= 1000 else 12 if votes <= 10000 else 18
        elif rating >= 5:
            return 5 if votes <= 1000 else 10 if votes <= 10000 else 15
        else:
            return 1 if votes <= 1000 else 2 if votes <= 10000 else 3

    df['Wa'] = df['actors'].apply(lambda actors: sum(Wa[actor] for actor in actors.split(',') if actor in Wa) if actors else 0)
    df['Wd'] = df['directors'].apply(lambda director: Wd[director] if director in Wd else 0)
    df['Wg'] = df['genres'].apply(lambda genres: sum(Wg[genre] for genre in genres.split(',') if genre in Wg) if genres else 0)
    df['Wy'] = df['startYear'].apply(lambda year: Wy[year] if year in Wy else 0)
    df['Wr'] = df.apply(calculate_wr, axis=1)
    df['Wm'] = df['Wa'] + df['Wd'] + df['Wg'] + df['Wy'] + df['Wr']
    return df

# Funzione per eseguire il clustering con K-means usando i centroidi manuali
def select_manual_centroids(df, n_clusters=4):
    features = df[['Wa', 'Wd', 'Wg', 'Wy', 'Wr']].values

    # Selezioniamo il primo centroide (ad esempio il primo film)
    centroids = [features[0]]  # Il primo centroide è il primo film
    
    # Seleziona i successivi centroidi in modo che siano distanziati dal precedente
    for _ in range(1, n_clusters):
        # Calcola la distanza Euclidea tra ogni punto e il centroide più vicino già selezionato
        distances = np.array([min([euclidean(f, c) for c in centroids]) for f in features])
        
        # Seleziona il punto che ha la distanza massima dal centroide più vicino
        next_centroid_index = np.argmax(distances)
        centroids.append(features[next_centroid_index])
    
    # Restituisci i centroidi selezionati
    return np.array(centroids)

# Funzione per eseguire il clustering con K-means usando i centroidi manuali distribuiti
def perform_kmeans_manual(df, n_clusters=4):
    features = df[['Wa', 'Wd', 'Wg', 'Wy', 'Wr']].values
    
    # Selezioniamo i centroidi manuali in modo bilanciato
    initial_centroids = select_manual_centroids(df, n_clusters)
    
    # Variabile per tracciare i cambiamenti nei centroidi
    prev_centroids = np.zeros_like(initial_centroids)
    
    # Itera fino a quando i centroidi non smettono di cambiare
    while not np.allclose(initial_centroids, prev_centroids):
        # Assegna ciascun film al centroide più vicino usando la distanza Euclidea
        df['cluster'] = pairwise_distances_argmin(features, initial_centroids)
        
        # Salva i centroidi precedenti
        prev_centroids = initial_centroids.copy()
        
        # Ricalcola nuovi centroidi come la media dei punti assegnati a ciascun cluster
        for i in range(n_clusters):
            # Otteniamo i film assegnati al cluster i
            cluster_points = features[df['cluster'] == i]
            # Se ci sono film assegnati al cluster, ricalcoliamo il centroide
            if len(cluster_points) > 0:
                initial_centroids[i] = cluster_points.mean(axis=0)
    print(df['cluster'].value_counts())
    # Restituiamo il dataframe con i cluster assegnati
    return df


def balance_clusters(df, min_cluster_size=5, max_cluster_size=10):
    # Conta il numero di film in ciascun cluster
    cluster_counts = df['cluster'].value_counts()

    # Trova i cluster troppo piccoli e quelli troppo grandi
    small_clusters = cluster_counts[cluster_counts < min_cluster_size].index.tolist()
    large_clusters = cluster_counts[cluster_counts > max_cluster_size].index.tolist()

    # Per i cluster troppo grandi, riassegna film ai cluster più piccoli
    for large_cluster in large_clusters:
        # Film da rimuovere dal cluster grande
        excess_movies = df[df['cluster'] == large_cluster].head(cluster_counts[large_cluster] - max_cluster_size)
        
        # Riassegna i film ai cluster piccoli
        for small_cluster in small_clusters:
            if len(excess_movies) == 0:
                break
            df.loc[excess_movies.index, 'cluster'] = small_cluster
            excess_movies = excess_movies.iloc[min_cluster_size:]  # Rimuovi i film già assegnati

    return df

def get_best_cluster_balanced(df, min_movies=5):
    # Calcola il rating medio per ciascun cluster
    cluster_ratings = df.groupby('cluster')['averageRating'].mean()
    print(cluster_ratings)
    # Trova il cluster con il rating medio più alto
    best_cluster = cluster_ratings.idxmax()

    # Controlla la dimensione del best cluster
    best_cluster_size = df[df['cluster'] == best_cluster].shape[0]

    # Se il best cluster ha meno di 5 film, aggiungi film da altri cluster vicini
    if best_cluster_size < min_movies:
        # Trova gli altri cluster ordinati per rating
        other_clusters = cluster_ratings.drop(best_cluster).sort_values(ascending=False)
        
        # Prendi film dai cluster con rating più alto finché non raggiungi 5 film
        for cluster in other_clusters.index:
            additional_movies = df[df['cluster'] == cluster].head(min_movies - best_cluster_size)
            df.loc[additional_movies.index, 'cluster'] = best_cluster
            best_cluster_size = df[df['cluster'] == best_cluster].shape[0]
            if best_cluster_size >= min_movies:
                break

    # Restituisci il best cluster con almeno 5 film
    return df[df['cluster'] == best_cluster]

# Funzione principale per la raccomandazione
def main(df, n_clusters=4):
    attributes = get_user_input(request.form)
    df = get_movies(df, attributes)
    if len(df) <= 20:
        return get_top_movies(df)
    df = get_top_movies(df)
    df = calculate_weights(df)
    df = perform_kmeans_manual(df, n_clusters)
    return get_best_cluster_balanced(df)

# Funzione per gestire l'input dell'utente dalla form di Flask
def get_user_input(form_data):
    attributes = {}
    start_year = form_data.get('startYear')
    end_year = form_data.get('endYear')
    if start_year or end_year:
        attributes['startYear'] = (start_year, end_year)
    genres = form_data.get('genres')
    if genres:
        attributes['genres'] = genres.split(',')
    actors = form_data.get('actor_names')
    if actors:
        attributes['actor_names'] = actors.split(',')
    directors = form_data.get('primaryName')
    if directors:
        attributes['primaryName'] = directors.split(',')
    return attributes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    df = pd.read_csv('preprocessed_data.csv')
    recommended_movies = main(df)
    return render_template('recommendations.html', movies=recommended_movies[['primaryTitle', 'averageRating', 'numVotes', 'startYear', 'cluster']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
