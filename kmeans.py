import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy as cp

'''
 ustawienia wizualizacji
'''
mpl.style.use('bmh')

'''
 przygotowanie danych
'''
n = 45
df = pd.read_csv('xclara.csv')
dane = df.sample(n)

x = dane['V1'].values
y = dane['V2'].values
f = np.array(list(zip(x, y)))
print(f)
plt.plot(x, y, '.')

'''
 grupowanie
 
 dist(a, b) - funkcja zwracająca długość odcinka |ab|
 k - liczba grup do odnalezienia
 C - zbiór punktów będących centrami grup
'''
k = 3
def dist(a, b, axis = 1):
    return np.linalg.norm(a - b, axis = axis)

# losowanie k startowych punktów centralnych
Cx = np.random.randint(0, np.max(f) - np.mean(f), size = k)
Cy = np.random.randint(0, np.max(f) - np.mean(f), size = k)
C = np.array(list(zip(Cx, Cy)), dtype = np.float32)
plt.plot(Cx, Cy, 'x', color = 'green', label = 'punkty startowe')

'''
 pętla główna
'''

## inicjalizacja zmiennych roboczych
Cprev = np.zeros(C.shape)
groups = np.zeros(n)
error = dist(C, Cprev, None)

## wykonuj pętlę dopóki odległość między wyznaczonymi punktami centralnymi jest różna
while error != 0:
    # dla każdej wartości przypisz najbliższą grupę
    for i in range(n):
        distances = dist(f[i], C)
        group = np.argmin(distances)
        groups[i] = group

    # skopiuj poprzednie punkty centralne
    Cprev = cp(C)

    # znajdź nowe punkty centralne obliczając średnią z każdej grupy
    for i in range(k):
        points = [f[j] for j in range(n) if groups[j] == i]
        C[i] = np.mean(points, axis = 0)

    # policz dystans między nowo wyznaczonymi punktami, a poprzednimi
    error = dist(C, Cprev, None)

plt.plot(C[:, 0], C[:, 1], '*', label = 'Punkty końcowe')
plt.legend()
plt.show()