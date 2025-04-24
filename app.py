import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request, send_file
import io
import base64
import os

app = Flask(__name__)

# Load and process data
def load_data():
    try:
        df = pd.read_excel("F1.xlsx")
        sheets = pd.read_excel("F1.xlsx", sheet_name=None)
        df_drivers = sheets["Driver Points"]
        df_prix = sheets["Champions -prix"]
        df_stats = sheets["Driver Stats"]
        df_constructor = sheets["Constructor points"]
        df_penalties = sheets["Penalty Points"]
        df_stats['Pos'] = df_stats['PTS'].rank(ascending=False, method='min').astype(int)
        print("df_stats columns:", df_stats.columns.tolist())  # Debug
        print("df_drivers columns:", df_drivers.columns.tolist())  # Debug
        return df, df_drivers, df_prix, df_stats, df_constructor, df_penalties, sheets
    except FileNotFoundError:
        print("Error: F1.xlsx not found")
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

df, df_drivers, df_prix, df_stats, df_constructor, df_penalties, sheets = load_data()

# F1 points system
f1_points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Function to encode plot to base64
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Cumulative Points Plot for Selected Drivers
def plot_cumulative_points(selected_drivers=None):
    if not selected_drivers:  # If None or empty list
        selected_drivers = ["Max Verstappen", "Lando Norris", "Charles Leclerc"]
    
    df_filtered = df_drivers[df_drivers["Driver"].isin(selected_drivers)]
    df_filtered = df_filtered.drop(columns=["P", "Pts"])
    df_long = df_filtered.melt(id_vars=["Driver"], var_name="Race", value_name="Position")
    df_long["Fastest_Lap"] = df_long["Position"].astype(str).str.contains(r"\*", regex=True)
    df_long["Position"] = df_long["Position"].astype(str).str.replace("*", "", regex=False)
    df_long["Position"] = pd.to_numeric(df_long["Position"], errors="coerce")
    df_long["Points"] = df_long["Position"].map(f1_points).fillna(0)
    df_long.loc[df_long["Fastest_Lap"] & (df_long["Points"] > 0), "Points"] += 1
    df_long["Cumulative Points"] = df_long.groupby("Driver")["Points"].cumsum()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_long, x="Race", y="Cumulative Points", hue="Driver", marker="o", linewidth=2.5)
    plt.title(f"Points Over Races: {', '.join(selected_drivers)}", fontsize=14)
    plt.xlabel("Race")
    plt.ylabel("Cumulative Points")
    plt.xticks(rotation=45)
    plt.grid(True)
    img_data = plot_to_base64()
    plt.close()
    return img_data

# Race Simulation and Probabilities
def run_simulation():
    num_racers = len(df_drivers)
    race_columns = df_drivers.columns[3:]
    num_races = len(race_columns)
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    min_prob_floor = 0.01
    fastest_lap_boost = 1.05
    dnf_penalty = 0.90
    consistency_bonus = 1.02

    race_results = np.zeros((num_races, num_racers), dtype=int)
    points = np.zeros(num_racers, dtype=int)
    win_probabilities = np.zeros((num_races, num_racers))
    cumulative_points = np.zeros((num_races, num_racers), dtype=int)
    top5_count = np.zeros(num_racers, dtype=int)
    win_probabilities[0] = np.ones(num_racers) / num_racers

    for race_idx, race in enumerate(race_columns):
        race_results_str = df_drivers[race].values
        ranking = np.zeros(num_racers, dtype=int)
        for racer_idx, result in enumerate(race_results_str):
            if pd.isna(result) or result in ["DNF", "WD", "DNS", "DQ"]:
                ranking[racer_idx] = num_racers
            else:
                result_cleaned = str(result).replace("*", "")
                try:
                    position = int(result_cleaned)
                    ranking[racer_idx] = position - 1
                    if position <= 5:
                        top5_count[racer_idx] += 1
                except ValueError:
                    ranking[racer_idx] = num_racers
        race_results[race_idx] = ranking + 1
        for racer, rank in enumerate(ranking):
            if (rank + 1) in points_system:
                points[racer] += points_system[rank + 1]
            if "*" in str(race_results_str[racer]) and (rank + 1) <= 10:
                points[racer] += 1
        cumulative_points[race_idx] = points.copy()
        if race_idx < num_races - 1:
            max_points_possible = (num_races - race_idx - 1) * 25
            championship_contenders = (points + max_points_possible >= max(points))
            contender_points = points[championship_contenders]
            if np.sum(contender_points) > 0:
                contender_prob = contender_points / np.sum(contender_points)
                contender_prob = np.maximum(contender_prob, min_prob_floor)
                for i, racer in enumerate(np.where(championship_contenders)[0]):
                    if "*" in str(race_results_str[racer]):
                        contender_prob[i] *= fastest_lap_boost
                    if race_results_str[racer] in ["DNF", "DNS", "DQ"]:
                        contender_prob[i] *= dnf_penalty
                    if top5_count[racer] >= 3:
                        contender_prob[i] *= consistency_bonus
                contender_prob /= np.sum(contender_prob)
            else:
                contender_prob = np.ones(len(contender_points)) / len(contender_points)
            win_prob = np.zeros(num_racers)
            win_prob[championship_contenders] = contender_prob
            win_probabilities[race_idx] = win_prob
        else:
            winner = np.argmax(points)
            win_probabilities[race_idx] = 0
            win_probabilities[race_idx, winner] = 1

    race_results_df = pd.DataFrame(race_results.T, columns=race_columns, index=df_drivers["Driver"])
    cumulative_points_df = pd.DataFrame(cumulative_points.T, columns=race_columns, index=df_drivers["Driver"])
    win_prob_df = pd.DataFrame(win_probabilities.T, columns=race_columns, index=df_drivers["Driver"])

    file_path = "Race_Results_and_Probabilities_Optimized.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        race_results_df.to_excel(writer, sheet_name="Race Results")
        cumulative_points_df.to_excel(writer, sheet_name="Cumulative Points")
        win_prob_df.to_excel(writer, sheet_name="Win Probabilities")
    
    return file_path

# Machine Learning Models
def train_ml_models():
    combined_data = pd.merge(df_stats, df_drivers, on="Driver")
    df_stats["Champion"] = (df_stats["PTS"] == df_stats["PTS"].max()).astype(int)
    X = df_stats[["1st", "Pod", "Pole", "FL", "RET"]]
    y = df_stats["Champion"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_pred, y_test, output_dict=True)
    
    importance = rf_model.feature_importances_
    feature_names = X.columns
    
    X_rf = df_stats[["1st", "Pod", "Pole"]]
    y_rf = df_stats["Champion"]
    rf_model_simple = RandomForestClassifier(random_state=42)
    rf_model_simple.fit(X_rf, y_rf)
    rf_importance = rf_model_simple.feature_importances_
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_rf, y_rf)
    dt_importance = dt_model.feature_importances_
    
    return accuracy, report, importance, feature_names, rf_importance, dt_importance

# Driver Analysis Functions
def analyze_driver1(driver_name):
    driver = df_stats[df_stats["Driver"] == driver_name]
    if driver.empty:
        return "Driver not found. Please enter a valid name."
    
    wins, podiums, poles, fastest_laps, dnfs, points = (
        driver["1st"].values[0], driver["Pod"].values[0], driver["Pole"].values[0],
        driver["FL"].values[0], driver["RET"].values[0], driver["PTS"].values[0]
    )
    champion = df_stats[df_stats["Champion"] == 1].iloc[0]
    champ_wins, champ_podiums, champ_poles, champ_fl, champ_dnfs, champ_points = (
        champion["1st"], champion["Pod"], champion["Pole"], champion["FL"], champion["RET"], champion["PTS"]
    )
    
    result = f"🏎 **Driver: {driver_name}**\n"
    result += f"- Wins: {wins} (Champion: {champ_wins})\n"
    result += f"- Podiums: {podiums} (Champion: {champ_podiums})\n"
    result += f"- Pole Positions: {poles} (Champion: {champ_poles})\n"
    result += f"- Fastest Laps: {fastest_laps} (Champion: {champ_fl})\n"
    result += f"- Retirements: {dnfs} (Champion: {champ_dnfs})\n"
    result += f"- Total Points: {points} (Champion: {champ_points})\n\n"
    
    if points == champ_points:
        result += "**🏆 Championship Winner!**\n"
        result += "✅ Dominated the season with strong performance across all metrics.\n"
    else:
        result += "**❌ Lost the Championship. Reasons:**\n"
        if wins == 0:
            result += f"❌ no wins\n"
        elif wins < champ_wins:
            result += f"⚠️ Fewer race wins ({wins} vs {champ_wins}) meant fewer high points-scoring finishes.\n"
        if podiums == 0:
            result += f"❌ no podiums\n"
        elif podiums < champ_podiums:
            result += f"⚠️ Fewer podiums ({podiums} vs {champ_podiums}) reduced consistency in scoring points.\n"
        if poles == 0:
            result += f"❌ no poles\n"
        elif poles < champ_poles:
            result += f"⚠️ Fewer pole positions ({poles} vs {champ_poles}) led to fewer race-winning opportunities.\n"
        if fastest_laps == 0:
            result += f"❌ no fastest laps\n"
        elif fastest_laps < champ_fl:
            result += f"⚠️ Fewer fastest laps ({fastest_laps} vs {champ_fl}) indicates slower race pace.\n"
        if dnfs > champ_dnfs:
            result += f"⚠️ More retirements ({dnfs} vs {champ_dnfs}) reduced total points potential.\n"
    
    _, _, _, _, rf_importance, _ = train_ml_models()
    result += "\n**ML Insights (Feature Importance):**\n"
    for feature, score in zip(["1st", "Pod", "Pole"], rf_importance):
        result += f"- {feature}: {score:.2f}\n"
    
    return result

def analyze_driver2(driver_name):
    combined_data = pd.merge(df_stats, df_drivers, on="Driver")
    driver = combined_data[combined_data["Driver"] == driver_name]
    race_columns = df_drivers.columns[3:]
    race_results = []
    for race in race_columns:
        if race in driver.columns:
            race_results.append(driver[race].values[0])
        else:
            race_results.append("NaN")
    
    positions = []
    for result in race_results:
        if pd.isna(result) or result == "NaN":
            positions.append(20)
        elif result in ["DNF", "DSQ", "WD", "DNS"]:
            positions.append(20)
        else:
            position = int(str(result).replace("*", ""))
            positions.append(position)
    
    best_race = race_columns[pd.Series(positions).idxmin()]
    worst_race = race_columns[pd.Series(positions).idxmax()]
    worst_result = race_results[pd.Series(positions).idxmax()]
    result = f"- Best Race: {best_race} (Position: {min(positions)})\n"
    result += f"- Worst Race: {worst_race} (Result: {worst_result})\n"
    
    recent_positions = positions[-5:] if len(positions) >= 5 else positions
    trend = "Improving" if recent_positions[-1] < recent_positions[0] else "Declining" if recent_positions[-1] > recent_positions[0] else "Stable"
    result += f"- Season Trend: {trend}\n"
    
    return result

# Top 3 Drivers Analysis Plots
def top3_analysis_plots():
    combined_data = pd.merge(df_stats, df_drivers, on="Driver")
    data_comparison = combined_data[combined_data["Driver"].isin(["Max Verstappen", "Lando Norris", "Charles Leclerc"])]
    data_comparison['Total_Podiums'] = data_comparison['1st'] + data_comparison['2nd'] + data_comparison['3rd']
    data_comparison['Win_Percentage'] = (data_comparison['1st'] / data_comparison['GP']) * 100
    data_comparison['Podium_Percentage'] = (data_comparison['Total_Podiums'] / data_comparison['GP']) * 100
    data_comparison['Race_Completion_Rate'] = ((data_comparison['GP'] - data_comparison['RET']) / data_comparison['GP']) * 100
    summary_stats = data_comparison[['Driver', 'GP', '1st', '2nd', '3rd', 'Total_Podiums', 'Pole', 'FL', 'RET', 'Win_Percentage', 'Podium_Percentage', 'Race_Completion_Rate']]
    
    plots = []
    
    # Win Percentage Bar Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_stats, x='Driver', y='Win_Percentage', palette="coolwarm", edgecolor="black")
    plt.xlabel("Driver", fontsize=14, fontweight='bold')
    plt.ylabel("Win Percentage", fontsize=14, fontweight='bold')
    plt.title("F1 Drivers' Win Percentage", fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    plots.append(plot_to_base64())
    plt.close()
    
    # Podium Percentage Bar Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_stats, x='Driver', y='Podium_Percentage', palette="coolwarm", edgecolor="black")
    plt.xlabel("Driver", fontsize=14, fontweight='bold')
    plt.ylabel("Podium Percentage", fontsize=14, fontweight='bold')
    plt.title("F1 Drivers' Podium Percentage", fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    plots.append(plot_to_base64())
    plt.close()
    
    # Podium Pie Charts
    colors = ['gold', 'silver', '#cd7f32']
    position = ['1st', '2nd', '3rd']
    podium_data = summary_stats[['Driver', '1st', '2nd', '3rd']]
    for driver in summary_stats["Driver"]:
        plt.figure(figsize=(6, 6))
        values = podium_data.loc[podium_data["Driver"] == driver, ["1st", "2nd", "3rd"]].values[0]
        wedges, texts, autotexts = plt.pie(values, labels=position, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
        plt.legend(wedges, position, title="Podium Positions", loc="upper right")
        plt.title(f"Podium Finishes for {driver}", fontsize=14, fontweight='bold')
        plots.append(plot_to_base64())
        plt.close()
    
    # Cumulative Points by Phase
    top_drivers = df_drivers[["Driver", "Pts"]].sort_values(by="Pts", ascending=False).head(3)["Driver"].tolist()
    df_filtered = df_drivers[df_drivers["Driver"].isin(top_drivers)]
    df_filtered = df_filtered.drop(columns=["P", "Pts"])
    df_long = df_filtered.melt(id_vars=["Driver"], var_name="Race", value_name="Position")
    df_long["Fastest_Lap"] = df_long["Position"].astype(str).str.contains(r"\*", regex=True)
    df_long["Position"] = df_long["Position"].astype(str).str.replace("*", "", regex=False)
    df_long["Position"] = pd.to_numeric(df_long["Position"], errors="coerce")
    df_long["Points"] = df_long["Position"].map(f1_points).fillna(0)
    df_long.loc[df_long["Fastest_Lap"] & (df_long["Points"] > 0), "Points"] += 1
    df_long["Cumulative Points"] = df_long.groupby("Driver")["Points"].cumsum()
    race_phases = {
        "Phase 1": df_drivers.columns[3:9],
        "Phase 2": df_drivers.columns[9:15],
        "Phase 3": df_drivers.columns[15:21],
        "Phase 4": df_drivers.columns[21:]
    }
    df_long["Phase"] = df_long["Race"].map({race: phase for phase, races in race_phases.items() for race in races})
    
    plt.figure(figsize=(12, 6))
    driver_palette = {"Max Verstappen": "blue", "Lando Norris": "orange", "Charles Leclerc": "green"}
    phase_styles = {"Phase 1": "solid", "Phase 2": "dashed", "Phase 3": "dotted", "Phase 4": "dashdot"}
    for driver in df_long["Driver"].unique():
        for phase in race_phases.keys():
            subset = df_long[(df_long["Driver"] == driver) & (df_long["Phase"] == phase)]
            plt.plot(
                subset["Race"], subset["Cumulative Points"],
                label=f"{driver} - {phase}",
                color=driver_palette[driver],
                linestyle=phase_styles[phase],
                marker="o", markersize=5, linewidth=2
            )
    plt.title("Cumulative Points Over Races (Split by Phases)", fontsize=14)
    plt.xlabel("Race")
    plt.ylabel("Cumulative Points")
    plt.xticks(rotation=45)
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1, 1))
    plots.append(plot_to_base64())
    plt.close()
    
    # Points Comparison Across Phases
    df_phasewise = df_long.groupby(["Driver", "Phase"])["Points"].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_phasewise, x="Phase", y="Points", hue="Driver", palette="tab10")
    plt.title("Points Comparison Across Phases", fontsize=14)
    plt.xlabel("Race Phases")
    plt.ylabel("Total Points in Phase")
    plt.legend(title="Driver")
    plots.append(plot_to_base64())
    plt.close()
    
    # Win Trend
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=data_comparison, x='GP', y='1st', hue='Driver', palette='viridis')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title('Win Trend Over the Season', fontsize=14)
    plt.xlabel('Grand Prix Number', fontsize=12)
    plt.ylabel('Number of Wins', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Driver', fontsize=10)
    plots.append(plot_to_base64())
    plt.close()
    
    # DNFs per Driver
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(data=data_comparison, x='Driver', y='RET', palette='coolwarm')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=12)
    plt.title('DNFs per Driver (Lower is Better)', fontsize=14)
    plt.xlabel('Driver', fontsize=12)
    plt.ylabel('Total DNFs', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plots.append(plot_to_base64())
    plt.close()
    
    return plots

# Single Route for All Content
@app.route('/', methods=['GET', 'POST'])
def index():
    all_drivers = df_drivers["Driver"].tolist()
    selected_drivers = request.form.getlist('drivers') if request.method == 'POST' else []
    cumulative_plot = plot_cumulative_points(selected_drivers)
    simulation_file = run_simulation()
    accuracy, report, importance, feature_names, rf_importance, dt_importance = train_ml_models()
    top3_plots = top3_analysis_plots()
    
    driver_name = None
    analysis1 = None
    analysis2 = None
    driver_position = None
    
    if request.method == 'POST':
        driver_name = request.form.get('driver_name')
        if driver_name:
            analysis1 = analyze_driver1(driver_name)
            analysis2 = analyze_driver2(driver_name)
            driver_row = df_drivers[df_drivers['Driver'] == driver_name]
            if not driver_row.empty:
                sorted_drivers = df_drivers.sort_values(by='Pts', ascending=False).reset_index()
                driver_position = sorted_drivers[sorted_drivers['Driver'] == driver_name].index[0] + 1 if driver_name in sorted_drivers['Driver'].values else 'N/A'
    
    feature_importance = list(zip(feature_names, importance))
    
    return render_template('index.html', 
                           cumulative_plot=cumulative_plot,
                           simulation_file=simulation_file,
                           accuracy=accuracy,
                           report=report,
                           importance=importance,
                           feature_names=feature_names,
                           rf_importance=rf_importance,
                           dt_importance=dt_importance,
                           top3_plots=top3_plots,
                           driver_name=driver_name,
                           analysis1=analysis1,
                           analysis2=analysis2,
                           all_drivers=all_drivers,
                           selected_drivers=selected_drivers,
                           df_drivers=df_drivers,
                           df_stats=df_stats,
                           df_constructor=df_constructor,
                           feature_importance=feature_importance,
                           driver_position=driver_position)

@app.route('/download_simulation')
def download_simulation():
    file_path = "Race_Results_and_Probabilities_Optimized.xlsx"
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
