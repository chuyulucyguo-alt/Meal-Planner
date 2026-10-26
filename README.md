# Weekly Meal Planner

An AI agent that plans a week of meals and a shopping list within your budget.
It's built as a single-file Streamlit app (`app.py`) that calls the `gpt-6-luna` model through [Portkey](https://portkey.ai).

## Features

- **Inputs:**
  - Ingredients you have, dietary restrictions or allergies, weekly budget (USD), and number of people.
  - **Days to plan:** 7 days, or 5 weekdays for meal prep.
  - **Meals per day:** any of breakfast, lunch, and dinner.
  - The menu, shopping list, budget, per-meal costs, and charts all follow your choices.
- **Nutrition balance:** shows how many planned meals include protein, vegetables, and whole grains. Python works this out by matching each dish's ingredients against food lists. The model isn't asked to guess.
- **Tool-calling agent** with two tools:
  - `estimate_cost(items)`: prices the shopping list from a hardcoded price table. All the math is done in Python, not by the model.
  - `check_allergens(recipe, allergies)`: flags dishes that contain your allergens. The agent replaces any unsafe dish.
- **Structured JSON output:** a 7-day menu (breakfast, lunch, dinner), a shopping list, and the total cost. The page shows them as menu cards and a priced table, and you can download the JSON.
- **Final verification:** after the agent finishes, the app re-checks every dish for allergens and recomputes the total in Python.
- **Safety:**
  - The input is checked for prompt-injection phrases (for example "ignore previous instructions" or "system prompt"). If one is found, the run is blocked.
  - Output tokens are capped for each model call, and total tokens are capped for each run.
  - Each session is limited to 10 runs.
  - The agent gets at most 5 rounds of tool calls.
- **Agent trace:** the "How the agent planned this" section lists every step in order: each model round, each tool call with its inputs and result, and whether unsafe dishes were replaced (for example "check_allergens: Pad Thai → contains peanut → replaced"). You can download the trace as Markdown.
- **"Use what you have" score:** a progress bar shows what share of your listed ingredients the plan uses.
- **Shopping list download:** a `.txt` checklist grouped by store section (Produce, Meat & Seafood, Dairy & Eggs, Pantry).
- **Charts (Altair):**
  - A budget journey chart shows each `estimate_cost` result across rounds, against a dashed budget line.
  - A donut chart breaks down the shopping list's cost by store section.
- **Visual summary:**
  - Stat cards show the total, the money left in your budget, the cost per meal, and the cost per person per day.
  - Each dish has tags for protein, veggie, and "your ingredients".
- **Warm theme:** a gradient header banner, with colors set in `.streamlit/config.toml`. There is also a "Try an example" button on the start screen.
- **Usage tracking:** the sidebar shows the input tokens, cached input tokens, output tokens, and estimated cost for each run. Every run, including blocked ones, is appended to `logs.csv`.

## Setup

1. **Requirements:** Python 3.10 or newer, and a Portkey API key with access to `gpt-6-luna`.

2. **Clone the repo and go into the project folder:**
   ```bash
   git clone <your-repo-url> Meal-Planner
   cd Meal-Planner
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Add your API key:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set `PORTKEY_API_KEY=your-key-here`.
   The app looks for `.env` in the project folder first, then in the parent folder.
   `.env` is listed in `.gitignore`, so it will never be committed.

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Then open **<http://localhost:8501>** in your browser.
   The app runs in headless mode (set in `.streamlit/config.toml`). It skips Streamlit's first-run email prompt, and it doesn't open a browser tab by itself.

## Usage

1. Enter the ingredients you already have, any restrictions or allergies (for example "vegetarian; allergic to peanuts"), your budget, and the number of people.
2. Click **Generate Plan**.
3. Look through the **Menu**, **Shopping list**, and **JSON** tabs. The sidebar shows token usage and cost for each run.

## Configuration

These constants are at the top of `app.py`:

| Setting | Default | Purpose |
|---|---|---|
| `MAX_COMPLETION_TOKENS` | 6000 | Output token cap for each model call |
| `MAX_TOKENS_PER_RUN` | 60000 | Total token cap for one run |
| `MAX_RUNS_PER_SESSION` | 10 | Runs allowed per browser session |
| `MAX_TOOL_ROUNDS` | 5 | Maximum rounds of tool calls |
| `PRICE_PER_1M_*` | placeholders | Model pricing used for the cost estimate. Update these to your real rates. |
| `PRICE_TABLE` | ~55 items | Grocery prices used by `estimate_cost` |

## Notes

- Through Portkey, `gpt-6-luna` requires `max_completion_tokens` instead of `max_tokens`. It also needs `reasoning_effort="none"` when function tools are used. The app already sets both.
- `logs.csv` is created on the first run and is listed in `.gitignore`.
