
# =====================================================
# PROJECT: DATA CLASSIFICATION AND REGRESSION
# =====================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")


# =====================================================
# STEP 2: IMPORT DATASET
# =====================================================

# Keep your CSV file in the same folder as this Python file

file_path = "Titanic Dataset.csv"

df = pd.read_csv(file_path)

# Rename Sex column to Gender
df.rename(columns={'Sex': 'Gender'}, inplace=True)

print("\nDataset imported successfully!")

print("\nFirst Five Rows:")
print(df.head())


# =====================================================
# STEP 3: EXPLORE DATASET
# =====================================================

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# =====================================================
# STEP 4: CHECK NULL VALUES
# =====================================================

print("\nNull Values Before Preprocessing:")
print(df.isnull().sum())

print("\nTotal Null Values:")
print(df.isnull().sum().sum())


# =====================================================
# STEP 5: DATA PREPROCESSING
# =====================================================

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows containing null values
df = df.dropna()

print("\nNull Values After Preprocessing:")
print(df.isnull().sum())

print("\nCleaned Dataset:")
print(df.head())

print("\nCleaned Dataset Shape:")
print(df.shape)


# =====================================================
# STEP 6: SURVIVED PEOPLE VS DEAD PEOPLE
# =====================================================

survival_counts = df['Survived'].value_counts()

dead = survival_counts.get(0, 0)
survived = survival_counts.get(1, 0)

plt.figure(figsize=(8, 5))

plt.bar(
    ['Dead', 'Survived'],
    [dead, survived],
    color=['red', 'green']
)

plt.title('Survived People vs Dead People')
plt.xlabel('Passenger Status')
plt.ylabel('Number of People')

plt.tight_layout()
plt.savefig("graph1_survival.png")
plt.show()


# =====================================================
# STEP 7: SURVIVED MALES VS DEAD MALES
# =====================================================

male_data = df[df['Gender'].str.lower() == 'male']

male_counts = male_data['Survived'].value_counts()

male_dead = male_counts.get(0, 0)
male_survived = male_counts.get(1, 0)

plt.figure(figsize=(8, 5))

plt.bar(
    ['Dead Males', 'Survived Males'],
    [male_dead, male_survived],
    color=['orange', 'blue']
)

plt.title('Male Survival Classification')
plt.xlabel('Gender')
plt.ylabel('Number of Males')

plt.tight_layout()
plt.savefig("graph2_male.png")
plt.show()


# =====================================================
# STEP 8: SURVIVED FEMALES VS DEAD FEMALES
# =====================================================

female_data = df[df['Gender'].str.lower() == 'female']

female_counts = female_data['Survived'].value_counts()

female_dead = female_counts.get(0, 0)
female_survived = female_counts.get(1, 0)

plt.figure(figsize=(8, 5))

plt.bar(
    ['Dead Females', 'Survived Females'],
    [female_dead, female_survived],
    color=['pink', 'purple']
)

plt.title('Female Survival Classification')
plt.xlabel('Gender')
plt.ylabel('Number of Females')

plt.tight_layout()
plt.savefig("graph3_female.png")
plt.show()


# =====================================================
# STEP 9: SURVIVED PEOPLE VS PASSENGER CLASS
# =====================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x='Pclass',
    hue='Survived',
    palette='Set2'
)

plt.title('Survival Classification by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of People')

plt.legend(
    title='Survival Status',
    labels=['Dead', 'Survived']
)

plt.tight_layout()
plt.savefig("graph4_class.png")
plt.show()


# =====================================================
# STEP 10: FINAL CLASSIFICATION RESULTS
# =====================================================

print("\n========== FINAL RESULTS ==========")

print("\nTotal Passengers:", len(df))

print("\nSurvival Details:")
print("Dead People:", dead)
print("Survived People:", survived)

print("\nMale Details:")
print("Dead Males:", male_dead)
print("Survived Males:", male_survived)

print("\nFemale Details:")
print("Dead Females:", female_dead)
print("Survived Females:", female_survived)

print("\nPassenger Class Details:")
print(df['Pclass'].value_counts().sort_index())

print("\n===================================")
print("DATA CLASSIFICATION PROJECT COMPLETED!")
print("===================================")
