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
Athlete = namedtuple("Athlete", ["athlete_number", "average_score"])
athlete_list = []

for i in range(athlete_avg.shape[0]):
    athlete_list.append(Athlete(i, athlete_avg[i]))

athlete_list_sorted = sorted(athlete_list, key=lambda x: x.average_score, reverse=True)

print("=" * 40)
for index in range(len(athlete_list_sorted)):
    athlete = athlete_list_sorted[index]
    print(f"Athlete at rank {index}: Athlete {athlete.athlete_number} with score {athlete.average_score}")
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