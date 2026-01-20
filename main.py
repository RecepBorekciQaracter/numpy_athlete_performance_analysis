# ========================
# 1. DATA & CONFIGURATION SECTION
# ========================
import numpy
from collections import namedtuple

np = numpy

SEPERATOR_LINE = "=" * 40

attributes = ["Strength","Speed","Stamina","Durability","Agility"]

weights = np.array([0.3, 0.15, 0.15, 0.3, 0.1])  # Power and Durability 

Athlete = namedtuple("Athlete", ["athlete_number", "average_score", "rank"], defaults=[None])
AthleteRanks = namedtuple("AthleteRanks", ["athlete_number", "raw_rank", "weighted_rank", "rank_change"])

# ========================
# 2. COMPUTATION SECTION
# ========================

# A. Load athlete data from file
def load_athlete_data(file_path):
    athlete_data = np.genfromtxt(file_path, delimiter=",")
    athlete_data = athlete_data.astype("int32")
    return athlete_data

# B. Basic Statistics by attribute
def compute_attribute_averages(athlete_data):
    attribute_avg = np.average(athlete_data, axis=0) # Average by attribute
    return attribute_avg
def compute_attribute_best_scores(athlete_data):
    attribute_max = np.max(athlete_data, axis=0) # Max by attribute
    return attribute_max
def compute_attribute_worst_scores(athlete_data):
    attribute_min = np.min(athlete_data, axis=0) # Min by attribute
    return attribute_min
def compute_attribute_consistency(athlete_data):
    attribute_std = np.std(athlete_data, axis=0) # Standard Deviation by attribute
    return attribute_std

# C. Performance metrics per athlete
def compute_athlete_averages(athlete_data):
    athlete_avg = np.average(athlete_data, axis=1) # Average by athlete
    return athlete_avg
def compute_athlete_best_scores(athlete_data):
    athlete_best = np.max(athlete_data, axis=1) # Max score by athlete
    return athlete_best
def compute_athlete_worst_scores(athlete_data):
    athlete_worst = np.min(athlete_data, axis=1) # Min score by athlete
    return athlete_worst
def compute_athlete_consistency(athlete_data):
    athlete_std = np.std(athlete_data, axis=1) # Standard Deviation by athlete
    return athlete_std

# D. Normalization (Apply min-max normalization)
def normalize_data_min_max(data):
    data_normalized = (data - data.min()) / (data.max() - data.min())
    return data_normalized

def compute_normalized_athlete_averages(athlete_data_normalized):
    athlete_avg_normalized = np.average(athlete_data_normalized, axis=1) # Normalized athlete averages
    return athlete_avg_normalized

# E. Weighted average
def compute_weighted_averages(athlete_data_normalized, weights):
    athlete_normalized_weighted_avg = np.average(athlete_data_normalized, weights=weights, axis=1) # Calculate weighted averages by athlete
    return athlete_normalized_weighted_avg

# F. Athlete Rankings
# Rank athletes by their average score
def rank_athletes_by_average(averages):
    athlete_list = []
    athlete_ranked_dict = dict()

    # Fill in the athlete list by creating Athlete objects.
    for i in range(averages.shape[0]):
        athlete_list.append(Athlete(i, averages[i]))

    # Sort athletes from highest to lowest by average score
    athlete_list_sorted = sorted(athlete_list, key=lambda x: x.average_score, reverse=True)

    # Add ranks to Athlete objects 
    # And collect these new athlete objects by filling the dict `athlete_ranked_dict` using (athlete_number, athlete) pairs.
    for index in range(len(athlete_list_sorted)):
        athlete = athlete_list_sorted[index]
        athlete = Athlete(
            athlete_number=athlete.athlete_number, 
            average_score=athlete.average_score, 
            rank=index + 1
        )
        athlete_ranked_dict[athlete.athlete_number] = athlete
    
    return athlete_ranked_dict

# G. Compare raw and weighted rankings
def compare_rankings(raw_ranked_dict, weighted_ranked_dict): 
    def validate_same_athletes(dict_a, dict_b):
        keys_a = set(dict_a.keys())
        keys_b = set(dict_b.keys())

        if keys_a != keys_b:
            missing_in_a = keys_b - keys_a
            missing_in_b = keys_a - keys_b

            raise ValueError(
                f"Athlete mismatch detected.\n"
                f"Missing in first dict: {missing_in_a}\n"
                f"Missing in second dict: {missing_in_b}"
            )

        return sorted(keys_a)

    athlete_comparison_list = []

    common_athletes = validate_same_athletes(raw_ranked_dict, weighted_ranked_dict)

    for athlete_number in common_athletes:
        athlete_ranked = raw_ranked_dict[athlete_number]
        athlete_weighted_ranked = weighted_ranked_dict[athlete_number]

        athlete_comparison = AthleteRanks(
            athlete_number=athlete_ranked.athlete_number,
            raw_rank=athlete_ranked.rank,
            weighted_rank=athlete_weighted_ranked.rank,
            rank_change=athlete_ranked.rank - athlete_weighted_ranked.rank
        )    

        athlete_comparison_list.append(athlete_comparison)

    return athlete_comparison_list

# ========================
# 3. REPORTING / PRESENTATION SECTION
# ========================


# Helper Functions
def print_separator():
    print(SEPERATOR_LINE)

def print_newline():
    print("\n")

# A. Print Raw Athlete Data
def print_athlete_data(attributes, athlete_data):
    print("ATHLETE DATA: ")
    print(attributes)
    print(athlete_data)
    print(athlete_data.shape)

# B. Basic Statistics By Attribute
# Average by attribute
def print_attribute_averages(attribute_avg):
    print("AVERAGE BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {attribute_avg[i]}")

# Max by attribute
def print_attribute_max_scores(attribute_max):
    print("MAX BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {attribute_max[i]}")

# Min by attribute
def print_attribute_min_scores(attribute_min):
    print("MIN BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {attribute_min[i]}")

# Standard Deviation by attribute
def print_attribute_standard_deviation(attribute_std):
    print("STANDART DEVIATION BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {round(attribute_std[i], 2)}")

# C. Basic Statistics by athlete
# Average by athlete
def print_athlete_averages(athlete_avg):
    print("AVERAGE BY ATHLETE: ")
    for i in range(athlete_avg.shape[0]):
        print(f"Athlete {i} Average: {athlete_avg[i]}")

# Max score by athlete
def print_athlete_max_scores(athlete_data, attributes):
    print("MAX SCORE BY ATHLETE: ")
    for i in range(athlete_data.shape[0]):
        print(f"Athlete {i} Best Score: {athlete_data[i]} in {attributes[np.argmax(athlete_data[i])]}")

# Min score by athlete
def print_athlete_min_scores(athlete_data, attributes):
    print("MIN SCORE BY ATHLETE: ")
    for i in range(athlete_data.shape[0]):
        print(f"Athlete {i} Worst Score: {athlete_data[i]} in {attributes[np.argmin(athlete_data[i])]}")

# Consistency by athlete
def print_athlete_standard_deviation(athlete_std):
    print("STANDARD DEVIATION OF PLAYERS: ")
    for i in range(athlete_std.shape[0]):
        print(f"Athlete {i} Standard Deviation: {round(athlete_std[i], 4)}")

# D. Normalization
# Print normalized data
def print_normalized_data(athlete_data_normalized):
    print("NORMALIZED ATTRIBUTES: ")
    print(athlete_data_normalized)

# Normalized athlete averages
def print_normalized_athlete_averages(athlete_avg_normalized):
    print("AVERAGE BY ATHLETE (NORMALIZED VALUES)")
    for i in range(athlete_avg_normalized.shape[0]):
        print(f"Athlete {i} Normalized Average: {round(athlete_avg_normalized[i], 4)}")

# E. Weighted Average
def print_weighted_averages_per_athlete(athlete_normalized_weighted_avg):
    print("WEIGHTED AVERAGE BY ATTRIBUTE PER ATHLETE (NORMALIZED VALUES): ")
    for i in range(athlete_normalized_weighted_avg.shape[0]):
        print(f"Athlete {i}: {round(athlete_normalized_weighted_avg[i], 4)}")

# F. Athlete rankings
# Rank athletes by their average score
def print_athlete_ranking_by_average(title, athlete_ranked_dict):
    print(title)
    for athlete_number in athlete_ranked_dict:
        athlete_ranked = athlete_ranked_dict[athlete_number]
        print(f"Athlete at rank {athlete_ranked.rank}: Athlete {athlete_ranked.athlete_number} with score {round(athlete_ranked.average_score, 4)}")

# G. Compare raw rankings and weighted rankings.
def print_comparison_of_rankings(athlete_comparison_list):
    print("COMPARISON OF RAW RANKINGS AND WEIGHTED RANKINGS: ")
    for athlete_comparison in athlete_comparison_list:
        print(f"Athlete {athlete_comparison.athlete_number}: " 
                f"Raw Rank = {athlete_comparison.raw_rank}, "
                f"Weighted Rank = {athlete_comparison.weighted_rank}, "
                f"Change = {"Increase by" if athlete_comparison.rank_change > 0 else "Decreased by" if athlete_comparison.rank_change < 0 else "No Change"} ({athlete_comparison.rank_change})"
        )

# ========================
# 4. MAIN FUNCTION
# ========================

def main():
    RAW_AVERAGE_TITLE = "ATHLETE RANKING BY RAW AVERAGE: "
    WEIGHTED_AVERAGE_TITLE = "ATHLETE RANKING BY WEIGHTED AVERAGE: "

    athlete_data = load_athlete_data("athlete_data.txt")
    athlete_avg = compute_athlete_averages(athlete_data)
    athlete_best = compute_athlete_best_scores(athlete_data)
    athlete_worst = compute_athlete_worst_scores(athlete_data)
    athlete_std = compute_athlete_consistency(athlete_data)

    attribute_avg = compute_attribute_averages(athlete_data)
    attribute_max = compute_attribute_best_scores(athlete_data)
    attribute_min = compute_attribute_worst_scores(athlete_data)
    attribute_std = compute_attribute_consistency(athlete_data)

    athlete_data_normalized = normalize_data_min_max(athlete_data)
    athlete_avg_normalized = compute_normalized_athlete_averages(athlete_data_normalized)

    athlete_normalized_weighted_avg = compute_weighted_averages(athlete_data_normalized, weights)

    athlete_ranked_dict = rank_athletes_by_average(athlete_avg) # Rank by raw averages
    athlete_weighted_ranked_dict = rank_athletes_by_average(athlete_normalized_weighted_avg) # Rank by normalized weighted averages

    athlete_comparison_list = compare_rankings(athlete_ranked_dict, athlete_weighted_ranked_dict)

    # A. Print Raw Data
    print_separator()
    print_athlete_data(attributes, athlete_data)
    print_separator()

    print_newline()

    # B. Print Basic Statistics by attribute
    # Average by attribute
    print_separator()
    print_attribute_averages(attribute_avg)
    print_separator()

    print_newline()

    # Max by attribute
    print_separator()
    print_attribute_max_scores(attribute_max)
    print_separator()

    print_newline()

    # Min by attribute
    print_separator()
    print_attribute_min_scores(attribute_min)
    print_separator()

    print_newline()

    # Standard Deviation by attribute
    print_separator()
    print_attribute_standard_deviation(attribute_std)
    print_separator()

    print_newline()

    # C. Print Basic Statistics by athlete
    # Average by athlete
    print_separator()
    print_athlete_averages(athlete_avg)
    print_separator()

    print_newline()

    # Max score by athlete
    print_separator()
    print_athlete_max_scores(athlete_data, attributes)
    print_separator()

    print_newline()

    # Min score by athlete
    print_separator()
    print_athlete_min_scores(athlete_data, attributes)
    print_separator()

    print_newline()

    # Consistency by athlete
    print_separator()
    print_athlete_standard_deviation(athlete_std)
    print_separator()

    print_newline()

    # D. Print normalized values
    print_separator()
    print_normalized_data(athlete_data_normalized)
    print_separator()

    print_newline()

    # Normalized athlete averages
    print_separator()
    print_normalized_athlete_averages(athlete_avg_normalized)
    print_separator()

    print_newline()

    # E. Print weighted averages
    print_separator()
    print_weighted_averages_per_athlete(athlete_normalized_weighted_avg)
    print_separator()

    print_newline()

    # F. Print rankings of athletes
    # Rank athletes by their average score
    print_separator()
    print_athlete_ranking_by_average(RAW_AVERAGE_TITLE, athlete_ranked_dict)
    print_separator()

    print_newline()

    # Rank athletes by weighted averages
    print_separator()
    print_athlete_ranking_by_average(WEIGHTED_AVERAGE_TITLE, athlete_weighted_ranked_dict)
    print_separator()

    print_newline()
    
    # G. Print Comparison of raw rankings and weighted rankings.
    print_separator()
    print_comparison_of_rankings(athlete_comparison_list)
    print_separator()

main()