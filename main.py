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
    """
    Load athlete performance data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing athlete data.
    
    Returns:
        np.ndarray: 2D array of athlete performance scores (athletes x attributes).
    """
    athlete_data = np.genfromtxt(file_path, delimiter=",")
    athlete_data = athlete_data.astype("int32")
    return athlete_data

# B. Basic Statistics by attribute
def compute_attribute_averages(athlete_data):
    """
    Calculate the average score for each attribute across all athletes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of average scores by attribute.
    """
    attribute_avg = np.average(athlete_data, axis=0) # Average by attribute
    return attribute_avg

def compute_attribute_best_scores(athlete_data):
    """
    Calculate the maximum score for each attribute across all athletes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of maximum scores by attribute.
    """
    attribute_max = np.max(athlete_data, axis=0) # Max by attribute
    return attribute_max

def compute_attribute_worst_scores(athlete_data):
    """
    Calculate the minimum score for each attribute across all athletes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of minimum scores by attribute.
    """
    attribute_min = np.min(athlete_data, axis=0) # Min by attribute
    return attribute_min

def compute_attribute_consistency(athlete_data):
    """
    Calculate the standard deviation of scores for each attribute across all athletes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of standard deviations by attribute.
    """
    attribute_std = np.std(athlete_data, axis=0) # Standard Deviation by attribute
    return attribute_std

# C. Performance metrics per athlete
def compute_athlete_averages(athlete_data):
    """
    Calculate the average score for each athlete across all attributes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of average scores by athlete.
    """
    athlete_avg = np.average(athlete_data, axis=1) # Average by athlete
    return athlete_avg

def compute_athlete_best_scores(athlete_data):
    """
    Calculate the best (maximum) score for each athlete across all attributes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of maximum scores by athlete.
    """
    athlete_best = np.max(athlete_data, axis=1) # Max score by athlete
    return athlete_best

def compute_athlete_worst_scores(athlete_data):
    """
    Calculate the worst (minimum) score for each athlete across all attributes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of minimum scores by athlete.
    """
    athlete_worst = np.min(athlete_data, axis=1) # Min score by athlete
    return athlete_worst

def compute_athlete_consistency(athlete_data):
    """
    Calculate the standard deviation of scores for each athlete across all attributes.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of standard deviations by athlete.
    """
    athlete_std = np.std(athlete_data, axis=1) # Standard Deviation by athlete
    return athlete_std

# D. Normalization (Apply min-max normalization)
def normalize_data_min_max(data):
    """
    Apply min-max normalization to scale data to the range [0, 1].
    
    Args:
        data (np.ndarray): Input array to normalize.
    
    Returns:
        np.ndarray: Normalized array with values between 0 and 1.
    """
    data_normalized = (data - data.min()) / (data.max() - data.min())
    return data_normalized

def compute_normalized_athlete_averages(athlete_data_normalized):
    """
    Calculate the average score for each athlete using normalized data.
    
    Args:
        athlete_data_normalized (np.ndarray): 2D array of normalized athlete performance scores.
    
    Returns:
        np.ndarray: 1D array of normalized athlete averages.
    """
    athlete_avg_normalized = np.average(athlete_data_normalized, axis=1) # Normalized athlete averages
    return athlete_avg_normalized

# E. Weighted average
def compute_weighted_averages(athlete_data_normalized, weights):
    """
    Calculate weighted average scores for each athlete using normalized data and attribute weights.
    
    Args:
        athlete_data_normalized (np.ndarray): 2D array of normalized athlete performance scores.
        weights (np.ndarray): 1D array of weights for each attribute.
    
    Returns:
        np.ndarray: 1D array of weighted average scores by athlete.
    """
    athlete_normalized_weighted_avg = np.average(athlete_data_normalized, weights=weights, axis=1) # Calculate weighted averages by athlete
    return athlete_normalized_weighted_avg

# F. Athlete Rankings
# Rank athletes by their average score
def rank_athletes_by_average(averages):
    """
    Rank athletes based on their average scores in descending order.
    
    Args:
        averages (np.ndarray): 1D array of average scores for each athlete.
    
    Returns:
        dict: Dictionary mapping athlete numbers to Athlete objects with ranks assigned.
    """
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
    """
    Compare raw and weighted rankings to identify changes in athlete positions.
    
    Args:
        raw_ranked_dict (dict): Dictionary of athletes ranked by raw averages.
        weighted_ranked_dict (dict): Dictionary of athletes ranked by weighted averages.
    
    Returns:
        list: List of AthleteRanks objects showing rank changes.
    
    Raises:
        ValueError: If the two dictionaries contain different sets of athletes.
    """
    def validate_same_athletes(dict_a, dict_b):
        """
        Validate that both dictionaries contain the same set of athletes.
        
        Args:
            dict_a (dict): First athlete ranking dictionary.
            dict_b (dict): Second athlete ranking dictionary.
        
        Returns:
            list: Sorted list of common athlete numbers.
        
        Raises:
            ValueError: If athlete sets differ between dictionaries.
        """
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
    """Print a separator line for visual organization of output."""
    print(SEPERATOR_LINE)

def print_newline():
    """Print a blank line for spacing in output."""
    print("\n")

# A. Print Raw Athlete Data
def print_athlete_data(attributes, athlete_data):
    """
    Print raw athlete performance data with attribute names and data shape.
    
    Args:
        attributes (list): List of attribute names.
        athlete_data (np.ndarray): 2D array of athlete performance scores.
    """
    print("ATHLETE DATA: ")
    print(attributes)
    print(athlete_data)
    print(athlete_data.shape)

# B. Basic Statistics By Attribute
def print_attribute_averages(attribute_avg):
    """
    Print average scores for each attribute.
    
    Args:
        attribute_avg (np.ndarray): 1D array of average scores by attribute.
    """
    print("AVERAGE BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {attribute_avg[i]}")

def print_attribute_max_scores(attribute_max):
    """
    Print maximum scores for each attribute.
    
    Args:
        attribute_max (np.ndarray): 1D array of maximum scores by attribute.
    """
    print("MAX BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {attribute_max[i]}")

def print_attribute_min_scores(attribute_min):
    """
    Print minimum scores for each attribute.
    
    Args:
        attribute_min (np.ndarray): 1D array of minimum scores by attribute.
    """
    print("MIN BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {attribute_min[i]}")

def print_attribute_standard_deviation(attribute_std):
    """
    Print standard deviation of scores for each attribute.
    
    Args:
        attribute_std (np.ndarray): 1D array of standard deviations by attribute.
    """
    print("STANDART DEVIATION BY ATTRIBUTE: ")
    for i in range(len(attributes)):
        print(f"{attributes[i]}: {round(attribute_std[i], 2)}")

# C. Basic Statistics by athlete
def print_athlete_averages(athlete_avg):
    """
    Print average scores for each athlete.
    
    Args:
        athlete_avg (np.ndarray): 1D array of average scores by athlete.
    """
    print("AVERAGE BY ATHLETE: ")
    for i in range(athlete_avg.shape[0]):
        print(f"Athlete {i} Average: {athlete_avg[i]}")

def print_athlete_max_scores(athlete_data, attributes):
    """
    Print the best score and corresponding attribute for each athlete.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
        attributes (list): List of attribute names.
    """
    print("MAX SCORE BY ATHLETE: ")
    for i in range(athlete_data.shape[0]):
        print(f"Athlete {i} Best Score: {athlete_data[i]} in {attributes[np.argmax(athlete_data[i])]}")

def print_athlete_min_scores(athlete_data, attributes):
    """
    Print the worst score and corresponding attribute for each athlete.
    
    Args:
        athlete_data (np.ndarray): 2D array of athlete performance scores.
        attributes (list): List of attribute names.
    """
    print("MIN SCORE BY ATHLETE: ")
    for i in range(athlete_data.shape[0]):
        print(f"Athlete {i} Worst Score: {athlete_data[i]} in {attributes[np.argmin(athlete_data[i])]}")

def print_athlete_standard_deviation(athlete_std):
    """
    Print standard deviation of scores for each athlete.
    
    Args:
        athlete_std (np.ndarray): 1D array of standard deviations by athlete.
    """
    print("STANDARD DEVIATION OF PLAYERS: ")
    for i in range(athlete_std.shape[0]):
        print(f"Athlete {i} Standard Deviation: {round(athlete_std[i], 4)}")

# D. Normalization
def print_normalized_data(athlete_data_normalized):
    """
    Print the normalized athlete performance data.
    
    Args:
        athlete_data_normalized (np.ndarray): 2D array of normalized athlete scores.
    """
    print("NORMALIZED ATTRIBUTES: ")
    print(athlete_data_normalized)

def print_normalized_athlete_averages(athlete_avg_normalized):
    """
    Print average scores for each athlete using normalized data.
    
    Args:
        athlete_avg_normalized (np.ndarray): 1D array of normalized average scores by athlete.
    """
    print("AVERAGE BY ATHLETE (NORMALIZED VALUES)")
    for i in range(athlete_avg_normalized.shape[0]):
        print(f"Athlete {i} Normalized Average: {round(athlete_avg_normalized[i], 4)}")

# E. Weighted Average
def print_weighted_averages_per_athlete(athlete_normalized_weighted_avg):
    """
    Print weighted average scores for each athlete using normalized data.
    
    Args:
        athlete_normalized_weighted_avg (np.ndarray): 1D array of weighted averages by athlete.
    """
    print("WEIGHTED AVERAGE BY ATTRIBUTE PER ATHLETE (NORMALIZED VALUES): ")
    for i in range(athlete_normalized_weighted_avg.shape[0]):
        print(f"Athlete {i}: {round(athlete_normalized_weighted_avg[i], 4)}")

# F. Athlete rankings
def print_athlete_ranking_by_average(title, athlete_ranked_dict):
    """
    Print athlete rankings in order by their average score.
    
    Args:
        title (str): Title to display for the ranking list.
        athlete_ranked_dict (dict): Dictionary of ranked athletes with their scores and ranks.
    """
    print(title)
    for athlete_number in athlete_ranked_dict:
        athlete_ranked = athlete_ranked_dict[athlete_number]
        print(f"Athlete at rank {athlete_ranked.rank}: Athlete {athlete_ranked.athlete_number} with score {round(athlete_ranked.average_score, 4)}")

# G. Compare raw rankings and weighted rankings.
def print_comparison_of_rankings(athlete_comparison_list):
    """
    Print comparison of raw and weighted rankings with rank changes for each athlete.
    
    Args:
        athlete_comparison_list (list): List of AthleteRanks objects with ranking comparisons.
    """
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
    """
    Main function that orchestrates the entire athlete performance analysis workflow.
    
    Performs the following operations:
    1. Loads athlete data from file
    2. Computes statistics (averages, max, min, standard deviation)
    3. Normalizes data and computes weighted averages
    4. Ranks athletes by raw and weighted averages
    5. Compares rankings and displays all results
    """
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