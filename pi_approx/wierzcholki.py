import numpy as np
import math
import matplotlib.pyplot as plt
import time


def obroc_punkt(punkt, kat):
    macierz_obrotu = np.array(
        [
            [np.cos(np.radians(kat)), -np.sin(np.radians(kat))],
            [np.sin(np.radians(kat)), np.cos(np.radians(kat))],
        ]
    )
    przekształcony_punkt = np.dot(macierz_obrotu, punkt)
    return przekształcony_punkt


def znajdz_wierzcholki(wierzcholek_poczatkowy, n):
    wierzcholki = [wierzcholek_poczatkowy]
    punkt = np.array(wierzcholek_poczatkowy)
    for _ in range(1, n):
        punkt = obroc_punkt(punkt, 360 / n)
        # punkt = np.round(punkt, 10)
        wierzcholki.append(punkt.tolist())
    return wierzcholki


punkt_poczatkowy = [0, 1]
n = 1000


## h2
def czy_suma_wektorow_zero(wierzcholki):
    suma_wektorow = np.zeros(2)
    for i in range(len(wierzcholki)):
        punkt1 = np.array(wierzcholki[i])
        punkt2 = np.array(wierzcholki[(i + 1) % len(wierzcholki)])
        wektor = punkt2 - punkt1
        suma_wektorow += wektor
    return suma_wektorow


wierzcholki = znajdz_wierzcholki(punkt_poczatkowy, n)
suma_wektorow = czy_suma_wektorow_zero(wierzcholki)

print("Suma wszystkich wektorów wi:", suma_wektorow)

## --

## h3


def czy_suma_wektorow_zero_h3(wierzcholki):
    suma_x_plus = 0
    suma_x_minus = 0
    suma_y_plus = 0
    suma_y_minus = 0

    x_plus = []
    x_minus = []
    y_plus = []
    y_minus = []

    for i in range(len(wierzcholki)):
        punkt1 = np.array(wierzcholki[i])
        punkt2 = np.array(wierzcholki[(i + 1) % len(wierzcholki)])
        wektor = punkt2 - punkt1

        if wektor[0] > 0:
            x_plus.append(wektor[0])
        else:
            x_minus.append(wektor[0])

        if wektor[1] > 0:
            y_plus.append(wektor[1])
        else:
            y_minus.append(wektor[1])

    x_plus.sort()
    x_minus.sort(reverse=True)
    y_plus.sort()
    y_minus.sort(reverse=True)

    suma_x_plus = sum(x_plus)
    suma_x_minus = sum(x_minus)
    suma_y_plus = sum(y_plus)
    suma_y_minus = sum(y_minus)

    suma_x = suma_x_plus + suma_x_minus
    suma_y = suma_y_plus + suma_y_minus

    return suma_x, suma_y


# Testujemy dla wielokąta o n=100
# punkt_poczatkowy = [0, 1]
# n = 100
wierzcholki = znajdz_wierzcholki(punkt_poczatkowy, n)
suma_x, suma_y = czy_suma_wektorow_zero_h3(wierzcholki)

print("Suma współrzędnych x:", suma_x)
print("Suma współrzędnych y:", suma_y)

## ---


def oblicz_obwod_po_wierzcholkach(wierzcholki):
    obwod = 0
    for i in range(len(wierzcholki)):
        punkt1 = np.array(wierzcholki[i])
        punkt2 = np.array(wierzcholki[(i + 1) % len(wierzcholki)])
        obwod += np.linalg.norm(punkt1 - punkt2)
    return obwod


wierzcholki = znajdz_wierzcholki(punkt_poczatkowy, n)


## h4
def find_iterations_for_accuracy():
    n = 10
    while True:
        punkt_poczatkowy = [0, 1]
        wierzcholki = znajdz_wierzcholki(punkt_poczatkowy, n)
        obwod = oblicz_obwod_po_wierzcholkach(wierzcholki)
        n += 1
        if round(obwod / 2, 5) == round(math.pi, 5):
            return n


# start_time = time.time()
# n_required = find_iterations_for_accuracy()
# execution_time = time.time() - start_time

# print(f"Liczba wierzchołków potrzebna dla dokładności 0.00001: {n_required}")
# print(f"Czas wykonania: {execution_time:.4f} sekundy")
## --


# print("Wielokąt ma", len(wierzcholki), "wierzchołków")
# print("Obwód wielokąta to:", oblicz_obwod_po_wierzcholkach(wierzcholki))
# print("Przybliżona wartość pi to:", oblicz_obwod_po_wierzcholkach(wierzcholki) / 2)

# n_test = [10, 100, 1000, 10000, 100000]
# n_test = np.geomspace(10, 10000, 100)
estimate_pi = []

# for n in n_test:
#     n = int(n)
#     wierzcholki = znajdz_wierzcholki(punkt_poczatkowy, n)
#     # print("Wielokąt ma", len(wierzcholki), "wierzchołków")
#     # print("Obwód wielokąta to:", oblicz_obwod_po_wierzcholkach(wierzcholki))
#     # print("Przybliżona wartość pi to:", oblicz_obwod_po_wierzcholkach(wierzcholki) / 2)
#     estimate_pi.append(oblicz_obwod_po_wierzcholkach(wierzcholki) / 2)
# print(
#     "Różnica między wartością pi a przybliżeniem to:",
#     math.pi - oblicz_obwod_po_wierzcholkach(wierzcholki) / 2,
# )
# print()

# print(estimate_pi)


# wykres podgladowy
def wykres_1():
    plt.plot(
        n_test,
        estimate_pi,
        marker="o",
        linestyle="--",
        color="b",
        label="Przybliżona wartość π",
    )
    plt.axhline(y=np.pi, color="r", linestyle="--", label="Dokładna wartość π")
    plt.xscale("log")  # Skala logarytmiczna dla osi x
    plt.ylim(3.141 - 0.01, 3.141 + 0.01)  # Rozciągnięcie osi y w pobliżu π
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Przybliżona wartość π")
    plt.title("Przybliżona wartość π w zależności od n")
    plt.legend()
    plt.grid(True)
    plt.show()


# wykres_1()


# wykres log bledu
def wykres_2():
    plt.plot(
        n_test,
        np.abs([est - math.pi for est in estimate_pi]),
        marker="o",
        linestyle="--",
        color="b",
        label="Błąd",
    )
    # plt.xscale("log")  # Skala logarytmiczna dla osi x
    plt.yscale("log")  # Skala logarytmiczna dla osi y
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Błąd")
    plt.title("Błąd w zależności od n")
    plt.legend()
    plt.grid(True)
    plt.show()


# wykres_2()

# zrobic wykres bledu (lograzmiczny n, liniowy blad)
