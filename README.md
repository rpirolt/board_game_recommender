# Board Game Recommender System

The application has been published and is available at: 
[https://gamebuddy.streamlit.app/
](https://gamescout.streamlit.app/)

## 🎲 Description
A hybrid recommendation system and interactive web app that suggests new board games based on player preferences, leveraging data from [BoardGameGeek](https://boardgamegeek.com/) and built with **Streamlit**, **Python**, and **machine learning**. The web-based UI surfaces board games recommendations by combining the powers of Collaborative Filtering (CF), Content-Based Filtering (CBF), and Large Language Models (LLMs). 

The software package is made up of several components, which work together to run the recommendation engine and front-end. The `data` folder houses the datasets used to train the models and power the app. Most of the data, such as user ratings and game attributes, were obtained from Kaggle. Additional attributes, including game descriptions, game mechanics, categories, types, player counts, and playtime, were obtained by scraping the BGG database via their API. The `data` folder also houses `precomputed_CBF.pkl`, which houses the data used for Content-Based Filtering (CBF) , as well as `V_final_quantized.npz`, which contains the item latent factor matrix for Collaborative Filtering. These files represent pre-calculated objects used by the CBF and CF-based predictions, respectively. 

The `notebooks` folder contains various Python notebooks that were used for data exploration, cleanup, and model training, etc. These files are not run when the app is launched. However, they contain important backround on how the models were built and what decisions were made in the process. For example, `cf.ipynb` was used to train the CF model and produce `V_final_quantized.npz`, which is used to predict user game ratings.

Lastly, the `src` folder houses the code that is run when the app is launched. `app.py` deploys, configures, and designs the streamlit app and captures inputs from the user. Each model component has it's own script (`cbf.py`, `cf.py`, and `llm.py`) which take the user inputs and generate scores for each of the 21k+ board games in the dataset. Then, `model_ensemble.py` combines the scores using a weighted average, applies filter logic to remove irrelevant results, and sends them back to the UI to be surfaced to the user. 

## 🧰 Installation Instructions
1. Clone this repo:
```bash
git clone https://github.com/rpirolt/board_game_recommender/
```
2.  Install dependencies
```bash
pip install -r requirements.txt
```
3. In your project root, create a folder named `.streamlit` (if it doesn’t exist), then create a file inside it called `secrets.toml`:
```bash
.streamlit/secrets.toml
```
Add your OpenAI API key inside:
```toml
OPENAI_API_KEY = "your_api_key_here"
```

## 🚀 Run the App
From the project root:
```bash
streamlit run src/app.py
```
Once the web app loads, use the sidebar to enter your board game preferences:

The available fields are as follows:

* **Liked Board Games**: Enter a few games you love — the more, the better!
* **Excluded From Recommendation**: Leave blank at first; you can use it later to filter out unwanted games.
* **Year Published, Minimum Rating, Player Count**: Optional filters — you can leave them as-is to start.
* **Play Time**: Try leaving it on Any the first time.
* **Weight (Complexity)**: Adjust if you want simpler or heavier games.
* **Game Mechanics**: Add your favorite mechanics (e.g., deck-building, area control).
* **Category/Theme**: Pick one, run the recommender, then try another to compare results.
* **Game Type**: Start with Strategy and/or Family Game — that covers most titles.

Click "Get Recommendations" to discover new boardgames, tailored to your preferences!

## 👩‍💻 Authors

**Team 43 — Georgia Tech**  
Chrissa da Gomez, Rene Pirolt, Bill Dvorkin, Evan Kai Hallberg, Elizabeth Kirk
