from flask import Flask, render_template, request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def recommend_movies_by_ids(movie_ids, top_n=5):
    try:
        # 選択された映画の評価ベクトル取得
        vectors = df_piv.loc[movie_ids].values

        # 平均ベクトル（仮想ユーザー）
        mean_vector = vectors.mean(axis=0).reshape(1, -1)

        # 類似映画検索
        distances, indices = rec_model.kneighbors(
            mean_vector,
            n_neighbors=top_n + len(movie_ids)
        )

        recommendations = []
        for idx in indices.flatten():
            movie_id = df_piv.index[idx]

            # 選択済み映画は除外
            if movie_id not in movie_ids:
                title = df_movies[df_movies["movie_id"] == movie_id]["movie_title"].values[0]
                recommendations.append(title)

            if len(recommendations) == top_n:
                break

        return recommendations

    except KeyError:
        return []



app = Flask(__name__)

# ===== データ読み込み =====
df = pd.read_csv("./ratings_100k.csv")
df = df.iloc[:, 0:3]

df_movies = pd.read_csv("./movies_100k.csv", sep="|")
df_movies.columns = ["movie_id", "movie_title"] + list(df_movies.columns[2:])

df_piv = df.pivot(index="movieId", columns="userId", values="rating").fillna(0)
df_sp = csr_matrix(df_piv.values)

rec_model = NearestNeighbors(
    n_neighbors=15,
    algorithm="brute",
    metric="cosine"
).fit(df_sp)

# ===== ルーティング =====
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        movies=df_movies[["movie_id", "movie_title"]].values
    )

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_ids = request.form.getlist("movie_ids")

    if len(movie_ids) != 3:
        return "映画を3つ選択してください"

    movie_ids = list(map(int, movie_ids))
    recommendations = recommend_movies_by_ids(movie_ids)

    return render_template(
        "result.html",
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)

