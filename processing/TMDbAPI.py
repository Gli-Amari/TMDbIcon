import requests
import pandas as pd

class TMDbAPI:
    def __init__(self, api_key):
        self.base_url = "https://api.themoviedb.org/3"
        self.api_key = api_key

    def get_movie_details(self, movie_id):
        endpoint = f"{self.base_url}/movie/{movie_id}"
        params = {
            "api_key": self.api_key
        }
        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_movies_list(self, endpoint, params = None):
        full_url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["api_key"] = self.api_key
        response = requests.get(full_url, params = params)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            return []

    def fetch_movie_data(self, endpoint, max_pages):
        data = []
        page = 1

        while page <= max_pages:
            params = {"page": page}
            movies_list = self.get_movies_list(endpoint, params)

            if not movies_list:
                break

            for movie in movies_list:
                movie_details = self.get_movie_details(movie["id"])
                if movie_details:
                    data.append({
                        "adult": movie_details.get("adult"),
                        "backdrop_path": movie_details.get("backdrop_path"),
                        "belongs_to_collection": movie_details.get("belongs_to_collection"),
                        "budget": movie_details.get("budget"),
                        "genres": movie_details.get("genres"),
                        "homepage": movie_details.get("homepage"),
                        "id": movie_details.get("id"),
                        "imdb_id": movie_details.get("imdb_id"),
                        "original_language": movie_details.get("original_language"),
                        "original_title": movie_details.get("original_title"),
                        "overview": movie_details.get("overview"),
                        "popularity": movie_details.get("popularity"),
                        "poster_path": movie_details.get("poster_path"),
                        "production_companies": movie_details.get("production_companies"),
                        "production_countries": movie_details.get("production_countries"),
                        "release_date": movie_details.get("release_date"),
                        "revenue": movie_details.get("revenue"),
                        "runtime": movie_details.get("runtime"),
                        "spoken_languages": movie_details.get("spoken_languages"),
                        "status": movie_details.get("status"),
                        "tagline": movie_details.get("tagline"),
                        "title": movie_details.get("title"),
                        "video": movie_details.get("video"),
                        "vote_average": movie_details.get("vote_average"),
                        "vote_count": movie_details.get("vote_count"),
                    })

            page += 1

        df = pd.DataFrame(data)

        # droppiamo le colonne che non servono a entrambe le KB
        column_to_delete = ["backdrop_path", "belongs_to_collection", "budget",
                            "poster_path", "video", "revenue", "homepage", "tagline",
                            "imdb_id"]

        df.drop(column_to_delete, axis = 1, inplace = True)
        return df
