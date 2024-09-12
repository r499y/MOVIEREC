# Movie Recommendation System

This repository contains a **Movie Recommendation System** that allows users to input their preferences (genres, actors, directors, release years, etc.) and receive personalized movie recommendations based on IMDb data. The system uses various techniques, including **K-means clustering** and **weighted attribute calculations**, to suggest top movies tailored to the user’s input.

## Features

- **User Preferences**: Users can input preferences such as genres, actor names, director names, and a range of release years.
- **Filtering**: The system filters the movie dataset based on user preferences.
- **Ranking and Sorting**: Movies are sorted based on their average rating and the number of votes they’ve received.
- **K-means Clustering**: The top 20 movies are clustered using K-means to ensure meaningful grouping.
- **Flask Web Interface**: A simple, user-friendly web interface where users can input their preferences and view recommendations.

## Getting Started

### Prerequisites

Before running the application, ensure you have the following installed:

- Flask (`pip install Flask`)
- pandas (`pip install pandas`)
- scikit-learn (`pip install scikit-learn`)
- Jupyter Notebook (to run `preprocess_data.ipynb`)

### Preparing the Dataset

The application requires a preprocessed dataset built from IMDb data. Follow these steps to download the datasets from IMDb and preprocess them into the format required by the application:

1. **Download IMDb Datasets**:

   Download the following datasets from the [IMDb website](https://datasets.imdbws.com):
   - `name.basics.tsv.gz`: [Link to IMDb name.basics](https://datasets.imdbws.com/name.basics.tsv.gz)
   - `title.ratings.tsv.gz`: [Link to IMDb title.ratings](https://datasets.imdbws.com/title.ratings.tsv.gz)
   - `title.crew.tsv.gz`: [Link to IMDb title.crew](https://datasets.imdbws.com/title.crew.tsv.gz)
   - `title.basics.tsv.gz`: [Link to IMDb title.basics](https://datasets.imdbws.com/title.basics.tsv.gz)
   - `title.principals.tsv.gz`: [Link to IMDb title.principals](https://datasets.imdbws.com/title.principals.tsv.gz)

   Extract the `.tsv.gz` files after downloading. Note that by clicking on each datasets links it will directly start the download of them

2. **Run the Preprocessing Notebook**:

   Open the `preprocess_data.ipynb` notebook in Jupyter and run the cells to preprocess the IMDb data and generate the final `preprocessed_data.csv` file, which will be used by the application.

   ```bash
   jupyter notebook preprocess_data.ipynb



## Running the Application

Once the dataset is prepared, follow these steps to run the movie recommendation system:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   ```

2. **Move to the project directory**:
   ```bash
   cd movie-recommendation-system
   ```

3. **Ensure the dataset (`preprocessed_data.csv`) is available**:
   After running the preprocessing notebook, ensure the `preprocessed_data.csv` file is located in the root folder of the project.

4. **Start the Flask server**:
   ```bash
   python app.py
   ```

5. **Access the web interface**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```


### Input your preferences:
- Fill in your preferred movie **genres**, **actors**, **directors**, and a range of **release years**.
- Click the "Get Recommendations" button.

### View recommendations: 
After submitting the form, the system will display a list of recommended movies on a new page, ranked by ratings and votes.
```

