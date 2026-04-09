import streamlit as st
import requests

BASE_URL = "https://movie-recommendation-system-6-75qk.onrender.com"

st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------- HEADER ----------------
st.title("🎬 Movie Recommender System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔎 Options")

category = st.sidebar.selectbox(
    "Select Category",
    ["popular", "top_rated", "upcoming", "now_playing", "trending"]
)

search_query = st.sidebar.text_input("Search Movie")

# ---------------- SAFE FETCH FUNCTIONS ----------------
def safe_request(url):
    try:
        res = requests.get(url, timeout=30)

        if res.status_code != 200:
            st.error("❌ Backend error")
            return None

        return res.json()

    except Exception as e:
     st.warning("⏳ Backend waking up... please wait 30-60 seconds")
    return None


def get_movies(category):
    url = f"{BASE_URL}/home?category={category}&limit=20"
    data = safe_request(url)

    if isinstance(data, list):
        return data
    return []


def search_movies(query):
    url = f"{BASE_URL}/tmdb/search?query={query}"
    data = safe_request(url)

    if isinstance(data, dict):
        return data.get("results", [])
    return []


def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/id/{movie_id}"
    data = safe_request(url)

    if isinstance(data, dict):
        return data
    return {}


# ---------------- MAIN LOGIC ----------------

movies = []

# 🔍 Search mode
if search_query:
    movies = search_movies(search_query)

# 🎬 Category mode
else:
    movies = get_movies(category)

# ---------------- DISPLAY MOVIES ----------------

st.subheader(
    f"🎥 Showing Movies: {category.upper() if not search_query else 'Search Results'}"
)

cols = st.columns(5)

for idx, movie in enumerate(movies):

    # 🔴 IMPORTANT FIX
    if not isinstance(movie, dict):
        continue

    with cols[idx % 5]:
        title = movie.get("title", "No Title")
        poster = movie.get("poster_path") or movie.get("poster_url")

        if poster:
            if "http" not in poster:
                poster = f"https://image.tmdb.org/t/p/w500{poster}"
            st.image(poster)

        if st.button(title, key=idx):
            st.session_state["movie_id"] = movie.get("id") or movie.get("tmdb_id")

# ---------------- MOVIE DETAILS ----------------

if "movie_id" in st.session_state:
    st.divider()
    st.subheader("🎬 Movie Details")

    details = get_movie_details(st.session_state["movie_id"])

    if not details:
        st.warning("No details found")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            if details.get("poster_url"):
                st.image(details["poster_url"])

        with col2:
            st.markdown(f"## {details.get('title', 'No Title')}")
            st.write(f"📅 Release Date: {details.get('release_date', 'N/A')}")
            st.write("⭐ Rating:", details.get("vote_average", "N/A"))
            st.write("📝 Overview:")
            st.write(details.get("overview", "No description"))

            # 🎯 Recommendations
            if st.button("Show Recommendations"):
                rec_url = f"{BASE_URL}/recommend/tfidf?title={details.get('title','')}"
                recs = safe_request(rec_url)

                if isinstance(recs, list):
                    st.subheader("🎯 Recommended Movies")
                    for r in recs:
                        st.write(f"👉 {r['title']} (Score: {round(r['score'],2)})")
                else:
                    st.warning("No recommendations available")