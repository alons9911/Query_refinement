import csv
import random
from enum import Enum

DATASET_SIZE = 200


class Education(Enum):
    UNDERGRADUATE = 1
    BACHELOR = 2
    MASTER = 3


class Gender(Enum):
    F = 1
    M = 2


def choose_education(gender: Gender):
    rand = random.random()
    if gender == Gender.M:
        if rand < 0.4:
            return Education.UNDERGRADUATE
        if rand < 0.7:
            return Education.BACHELOR
        return Education.MASTER
    else:
        if rand < 0.5:
            return Education.UNDERGRADUATE
        if rand < 0.8:
            return Education.BACHELOR
        return Education.MASTER


def choose_age():
    return random.randint(20, 50)


def choose_gender():
    return random.choice([Gender.M, Gender.F])


def choose_income(gender: Gender, education: Education, age: int):
    chance = 0.5
    if gender == gender.M:
        chance += 0.15
    else:
        chance -= 0.1
    if education == education.MASTER:
        chance += 0.1
    elif education == education.UNDERGRADUATE:
        chance -= 0.1
    chance += (age - 35) / 100

    rand = random.random()
    if rand < chance:
        return 1
    return 0


def main():
    adults = []
    for i in range(DATASET_SIZE):
        gender = choose_gender()
        education = choose_education(gender)
        age = choose_age()

        income = choose_income(gender, education, age)
        adults.append((age, education.name, gender.name, income, 'yes'))
    with open('syn_adults.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['age', 'education', 'gender', 'income', 'all'])
        for row in adults:
            writer.writerow(row)


if __name__ == '__main__':
    main()
