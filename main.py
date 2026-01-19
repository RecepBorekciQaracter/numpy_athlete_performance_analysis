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

athlete_list = []
athlete_ranked_list = []

athlete_weighted_avg_list = []
athlete_weighted_ranked_list = []

# ========================
# 2. COMPUTATION SECTION
# ========================

athlete_data = np.genfromtxt("athlete_data.txt", delimiter=",")
athlete_data = athlete_data.astype("int32")

# A. Performance metrics
athlete_avg = np.average(athlete_data, axis=1) # Average by athlete
athlete_best = np.max(athlete_data, axis=1) # Max score by athlete
athlete_worst = np.min(athlete_data, axis=1) # Min score by athlete
athlete_std = np.std(athlete_data, axis=1) # Standart Deviation by athlete

# B. Basic Statistics by attribute
attribute_avg = np.average(athlete_data, axis=0) # Average by attribute
attribute_max = np.max(athlete_data, axis=0) # Max by attribute
attribute_min = np.min(athlete_data, axis=0) # Min by attribute
attribute_std = np.std(athlete_data, axis=0) # Standart Deviation by attribute

# C. Athlete Ranking by raw average

# Fill in the athlete list by creating Athlete objects.
for i in range(athlete_avg.shape[0]):
    athlete_list.append(Athlete(i, athlete_avg[i]))

# Sort athletes from highest to lowest by average score
athlete_list_sorted = sorted(athlete_list, key=lambda x: x.average_score, reverse=True)

# Add ranks to Athlete objects and fill the list `athlete_ranked_list`.
for index in range(len(athlete_list_sorted)):
    athlete = athlete_list_sorted[index]
    athlete = Athlete(
        athlete_number=athlete.athlete_number, 
        average_score=athlete.average_score, 
        rank=index + 1
    )
    athlete_ranked_list.append(athlete)

# D. Normalization (Apply min-max normalization)
athlete_data_normalized = (athlete_data - athlete_data.min()) / (athlete_data.max() - athlete_data.min())

athlete_avg_normalized = np.average(athlete_data_normalized, axis=1) # Normalized athlete averages

# E. Weighted average
athlete_normalized_weighted_avg = np.average(athlete_data_normalized, weights=weights, axis=1) # Calculate weighted averages by athlete

# Fill the weighted average list using Athlete objects.
for i in range(athlete_normalized_weighted_avg.shape[0]):
    athlete_weighted_avg_list.append(Athlete(i, athlete_normalized_weighted_avg[i]))

# F. Athlete ranking by weighted averages.
# Sort athletes by their weighted average of their normalized values
athlete_weighted_avg_list_sorted = sorted(athlete_weighted_avg_list, key=lambda x: x.average_score, reverse=True)

# Append ranks to the Athlete objects. And collect these new athlete objects in the list `athlete_weighted_ranked_list`
for index in range(len(athlete_weighted_avg_list_sorted)):
    athlete = athlete_weighted_avg_list_sorted[index]
    athlete = Athlete(
        athlete_number=athlete.athlete_number, 
        average_score=athlete.average_score, 
        rank=index + 1
    )
    athlete_weighted_ranked_list.append(athlete)

# ========================
# 3. REPORTING / PRESENTATION SECTION
# ========================

print(SEPERATOR_LINE)
print("ATHLETE DATA: ")
print(attributes)
print(athlete_data)
print(SEPERATOR_LINE)
print("\n")

# Average by attribute
print(SEPERATOR_LINE)
print("AVERAGE BY ATTRIBUTE: ")
for i in range(len(attributes)):
    print(f"{attributes[i]}: {attribute_avg[i]}")
print(SEPERATOR_LINE)

print("\n")

# Average by athlete

print(SEPERATOR_LINE)
print("AVERAGE BY ATHLETE: ")
for i in range(athlete_avg.shape[0]):
    print(f"Athlete {i} Average: {athlete_avg[i]}")
print(SEPERATOR_LINE)

print("\n")

# Max score by athlete

print(SEPERATOR_LINE)
print("MAX SCORE BY ATHLETE: ")
for i in range(athlete_best.shape[0]):
    print(f"Athlete {i} Best Score: {athlete_best[i]} in {attributes[np.argmax(athlete_data[i])]}")
print(SEPERATOR_LINE)

print("\n")

# Min score by athlete

print(SEPERATOR_LINE)
print("MIN SCORE BY ATHLETE: ")
for i in range(athlete_worst.shape[0]):
    print(f"Athlete {i} Worst Score: {athlete_worst[i]} in {attributes[np.argmin(athlete_data[i])]}")
print(SEPERATOR_LINE)

print("\n")

# Consistency by athlete

print(SEPERATOR_LINE)
print("STANDARD DEVIATION OF PLAYERS: ")
for i in range(athlete_std.shape[0]):
    print(f"Athlete {i} Standard Deviation: {round(athlete_std[i], 4)}")
print(SEPERATOR_LINE)

print("\n")

print(athlete_list)
print(athlete_list_sorted)
print(athlete_ranked_list)

# Rank athletes by their average score
print(SEPERATOR_LINE)
for index, athlete_ranked in enumerate(athlete_ranked_list):
    print(f"Athlete at rank {athlete_ranked.rank}: Athlete {athlete_ranked.athlete_number} with score {athlete_ranked.average_score}")
print(SEPERATOR_LINE)

print("\n")

# Basic Statistics by attribute
# Max by attribute
print(SEPERATOR_LINE)
print("MAX BY ATTRIBUTE: ")
for i in range(len(attributes)):
    print(f"{attributes[i]}: {attribute_max[i]}")
print(SEPERATOR_LINE)

print("\n")

# Min by attribute
print(SEPERATOR_LINE)
print("MIN BY ATTRIBUTE: ")
for i in range(len(attributes)):
    print(f"{attributes[i]}: {attribute_min[i]}")
print(SEPERATOR_LINE)

print("\n")

# Standart Deviation by attribute
print(SEPERATOR_LINE)
print("STANDART DEVIATION BY ATTRIBUTE: ")
for i in range(len(attributes)):
    print(f"{attributes[i]}: {round(attribute_std[i], 2)}")
print(SEPERATOR_LINE)

print("\n")

# NORMALIZATION
print(SEPERATOR_LINE)
print("Normalized attributes: ")
print(athlete_data_normalized)

print(SEPERATOR_LINE)
print("\n")
print(SEPERATOR_LINE)

# Normalized athlete averages
print("AVERAGE BY ATHLETE (NORMALIZED VALUES)")
for i in range(athlete_avg_normalized.shape[0]):
    print(f"Athlete {i} Normalized Average: {round(athlete_avg_normalized[i], 4)}")

print(SEPERATOR_LINE)

print("\n")

# WEIGHTED AVERAGE

print(SEPERATOR_LINE)
print("WEIGHTED AVERAGE BY ATTRIBUTE PER ATHLETE (NORMALIZED VALUES): ")
for i in range(athlete_normalized_weighted_avg.shape[0]):
    print(f"Athlete {i}: {round(athlete_normalized_weighted_avg[i], 4)}")
print(SEPERATOR_LINE)

print("\n")

# Rank athletes by weighted averages
print(SEPERATOR_LINE)
print("ATHLETE RANKING BY WEIGHTED AVERAGE: ")

for index, athlete_ranked in enumerate(athlete_weighted_ranked_list):
    print(f"Athlete at rank {athlete_ranked.rank}: Athlete {athlete_ranked.athlete_number} with weighted score {round(athlete_ranked.average_score, 4)}")

print(SEPERATOR_LINE)

print("\n")

print(SEPERATOR_LINE)

# Compare raw rankings and weighted rankings.
print("COMPARISON OF RAW RANKINGS AND WEIGHTED RANKINGS: ")
for index, athete_ranked in enumerate(athlete_ranked_list):
    for jndex, athlete_weighted_ranked in enumerate(athlete_weighted_ranked_list):
        if athete_ranked.athlete_number == athlete_weighted_ranked.athlete_number:
            athlete_comparison = AthleteRanks(
                athlete_number=athete_ranked.athlete_number,
                raw_rank=athete_ranked.rank,
                weighted_rank=athlete_weighted_ranked.rank,
                rank_change=athete_ranked.rank - athlete_weighted_ranked.rank
            )
            print(f"Athlete {athlete_comparison.athlete_number}: Raw Rank = {athlete_comparison.raw_rank}, Weighted Rank = {athlete_comparison.weighted_rank}, Change = {"Increase by" if athlete_comparison.rank_change > 0 else "Decreased by" if athlete_comparison.rank_change < 0 else "No Change"} ({athlete_comparison.rank_change})")

print(SEPERATOR_LINE)