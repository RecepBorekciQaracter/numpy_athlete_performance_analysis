import numpy
from collections import namedtuple

np = numpy

# Attributes
attributes = ["Strength","Speed","Stamina","Durability","Agility"]

athlete_data = np.genfromtxt("athlete_data.txt", delimiter=",")
athlete_data = athlete_data.astype("int32")

print(athlete_data)

print("\n")

# Average by attribute
attribute_avg = np.average(athlete_data, axis=0)
print("=" * 40)
print("AVERAGE BY ATTRIBUTE: ")
for i in range(len(attributes)):
    print(f"{attributes[i]}: {attribute_avg[i]}")
print("=" * 40)

print("\n")

# Average by athlete
athlete_avg = np.average(athlete_data, axis=1)

print("=" * 40)
print("AVERAGE BY ATHLETE: ")
for i in range(athlete_avg.shape[0]):
    print(f"Athlete {i} Average: {athlete_avg[i]}")
print("=" * 40)

print("\n")

# Max score by athlete
athlete_best = np.max(athlete_data, axis=1)
print(athlete_best)
print("=" * 40)
print("MAX SCORE BY ATHLETE: ")
for i in range(athlete_best.shape[0]):
    print(f"Athlete {i} Best Score: {athlete_best[i]} in {attributes[np.argmax(athlete_data[i])]}")
print("=" * 40)

print("\n")

# Min score by athlete
athlete_worst = np.min(athlete_data, axis=1)
print(athlete_worst)
print("=" * 40)
print("MIN SCORE BY ATHLETE: ")
for i in range(athlete_worst.shape[0]):
    print(f"Athlete {i} Worst Score: {athlete_worst[i]} in {attributes[np.argmin(athlete_data[i])]}")
print("=" * 40)

print("\n")

# Consistency by athlete
athelete_std = np.std(athlete_data, axis=1)

print(athelete_std)
print("=" * 40)
print("STANDARD DEVIATION OF PLAYERS: ")
for i in range(athelete_std.shape[0]):
    print(f"Athlete {i} Standard Deviation: {athelete_std[i]}")
print("=" * 40)

print("\n")

# Rank athletes by their average score
Athlete = namedtuple("Athlete", ["athlete_number", "average_score", "rank"], defaults=[None])
athlete_list = []
athlete_ranked_list = []

for i in range(athlete_avg.shape[0]):
    athlete_list.append(Athlete(i, athlete_avg[i]))

athlete_list_sorted = sorted(athlete_list, key=lambda x: x.average_score, reverse=True)

print("=" * 40)
for index in range(len(athlete_list_sorted)):
    athlete = athlete_list_sorted[index]
    athlete = Athlete(
        athlete_number=athlete.athlete_number, 
        average_score=athlete.average_score, 
        rank=index + 1
    )
    athlete_ranked_list.append(athlete)
    print(f"Athlete at rank {athlete.rank}: Athlete {athlete.athlete_number} with score {athlete.average_score}")
print("=" * 40)

print("\n")

# Basic Statistics by attribute
# Max by attribute
print("=" * 40)
print("MAX BY ATTRIBUTE: ")
attribue_max = np.max(athlete_data, axis=0)
for i in range(len(attributes)):
    print(f"{attributes[i]}: {attribue_max[i]}")
print("=" * 40)

print("\n")

# Min by attribute
print("=" * 40)
print("MIN BY ATTRIBUTE: ")
attribue_min = np.min(athlete_data, axis=0)
for i in range(len(attributes)):
    print(f"{attributes[i]}: {attribue_min[i]}")
print("=" * 40)

print("\n")

# Standart Deviation by attribute
print("=" * 40)
print("STANDART DEVIATION BY ATTRIBUTE: ")
attribue_std = np.std(athlete_data, axis=0)
for i in range(len(attributes)):
    print(f"{attributes[i]}: {round(attribue_std[i], 2)}")
print("=" * 40)

print("\n")

# NORMALIZATION
athlete_data_normalized = (athlete_data - athlete_data.min()) / (athlete_data.max() - athlete_data.min())

print("=" * 40)
print("Normalized attributes: ")
print(athlete_data_normalized)

print("=" * 40)
print("\n")
print("=" * 40)

athelete_avg_normalized = np.average(athlete_data_normalized, axis=1)
print("AVERAGE BY ATHLETE (NORMALIZED VALUES)")
for i in range(athelete_avg_normalized.shape[0]):
    print(f"Athlete {i} Normalized Average: {round(athelete_avg_normalized[i], 4)}")

print("=" * 40)

print("\n")

# WEIGHTED AVERAGE
weights = np.array([0.3, 0.15, 0.15, 0.3, 0.1])  # Power and Durability 

athlete_normalized_weighted_avg = np.average(athlete_data_normalized, weights=weights, axis=1)

athlete_weighted_avg_list = []
athlete_weighted_ranked_list = []

print("=" * 40)
print("WEIGHTED AVERAGE BY ATTRIBUTE PER ATHLETE (NORMALIZED VALUES): ")
for i in range(athlete_normalized_weighted_avg.shape[0]):
    print(f"Athlete {i}: {round(athlete_normalized_weighted_avg[i], 4)}")
    athlete_weighted_avg_list.append(Athlete(i, athlete_normalized_weighted_avg[i]))
print("=" * 40)

print("\n")

# Rank athletes by weighted averages
print("=" * 40)
print("ATHLETE RANKING BY WEIGHTED AVERAGE: ")
athlete_weighted_avg_list_sorted = sorted(athlete_weighted_avg_list, key=lambda x: x.average_score, reverse=True)

for index in range(len(athlete_weighted_avg_list_sorted)):
    athlete = athlete_weighted_avg_list_sorted[index]
    athlete = Athlete(
        athlete_number=athlete.athlete_number, 
        average_score=athlete.average_score, 
        rank=index + 1
    )
    athlete_weighted_ranked_list.append(athlete)
    print(f"Athlete at rank {athlete.rank}: Athlete {athlete.athlete_number} with weighted score {round(athlete.average_score, 4)}")

print("=" * 40)

print("\n")

print("=" * 40)

# Compare raw rankings and weighted rankings.
print("COMPARISON OF RAW RANKINGS AND WEIGHTED RANKINGS: ")
AthleteRanks = namedtuple("AthleteRanks", ["athlete_number", "raw_rank", "weighted_rank", "rank_change"])

for index in range(len(athlete_ranked_list)):
    athete_ranked = athlete_ranked_list[index]

    for jndex in range(len(athlete_weighted_ranked_list)):
        athlete_weighted_ranked = athlete_weighted_ranked_list[jndex]
        if athete_ranked.athlete_number == athlete_weighted_ranked.athlete_number:
            athlete_comparison = AthleteRanks(
                athlete_number=athete_ranked.athlete_number,
                raw_rank=athete_ranked.rank,
                weighted_rank=athlete_weighted_ranked.rank,
                rank_change=athete_ranked.rank - athlete_weighted_ranked.rank
            )
            print(f"Athlete {athlete_comparison.athlete_number}: Raw Rank = {athlete_comparison.raw_rank}, Weighted Rank = {athlete_comparison.weighted_rank}, Change = {"Increase by" if athlete_comparison.rank_change > 0 else "Decreased by" if athlete_comparison.rank_change < 0 else "No Change"} ({athlete_comparison.rank_change})")

print("=" * 40)