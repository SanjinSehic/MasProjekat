import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearnaRegresija:

    # formule za racunanje koeficijenata pronadjene na internetu
    def procjeni_koeficijente(self, x, y):
        n = np.size(x)

        m_x = np.mean(x)
        m_y = np.mean(y)

        SS_xy = np.sum(y * x) - n * m_y * m_x
        SS_xx = np.sum(x * x) - n * m_x * m_x

        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1 * m_x

        return (b_0, b_1)

# funkcija za pokretanje i crtanje linearne regresije
def pokreniLinearnu(X,Y):
    # djele se podaci za treniranje i testiranje ( 80% treniranje i 20% testiranje)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=125)

    linearna = LinearnaRegresija()
    b = linearna.procjeni_koeficijente(X_train, Y_train)

    # promjeniti odstupanje po potrebi
    odstupanje = 10500

    print("-------------------------------------------------------------------")
    print("Koeficijeti linearne regresije su:", b)
    print("Pouzdanost linearne regresije sa odstupanjem", odstupanje, "je ",
          pouzdanost(Y_test, b[1] * X_test + b[0], odstupanje) * 100, " posto")
    print("Ukupna greska linearne regresije je", ukupna_greska(Y_test, b[1] * X_test + b[0]))
    print("-------------------------------------------------------------------")

    # crtanje linearne regresije
    plt.scatter(X, Y, color="b", marker="s", s=30)
    prava = b[0] + b[1] * X_test
    plt.plot(X_test, prava, color="r")
    plt.xlabel('Godine iskustva')
    plt.ylabel('Plata')
    plt.title('Linearna regresija')
    plt.show()


class RidgeRegresija:
    # preciznost zavisi od nivoa ucenja i broja iteracija
    def __init__(self, nivo_ucenja=0.01, iteracija=10000, regularizacija=1):
        self.nivo_ucenja = nivo_ucenja
        self.iteracija = iteracija
        self.regularizacija = regularizacija

    # formule za racunanje koeficijenata pronadjene na internetu
    def procjeni_koeficijente(self, X, Y):

        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iteracija):
            self.azuriraj()
        return self

    # pozivamo funkciju onoliko puta koliki je broj iteracija
    def azuriraj(self):
        Y_pred = self.procijeni(self.X)

            # racunanje gradienta
        dW = (- (2 * (self.X.T).dot(self.Y - Y_pred)) +
              (2 * self.regularizacija * self.W)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m

            # azuriraj
        self.W = self.W - self.nivo_ucenja * dW
        self.b = self.b - self.nivo_ucenja * db
        return self

    # procjenjuje y ( x*w + b)
    def procijeni(self, X):
        return X.dot(self.W) + self.b


# funkcija za pokretanje i crtanje ridge regresije
def pokreniRidge(X,Y):
    # djele se podaci za treniranje i testiranje ( 80% treniranje i 20% testiranje)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=125)

    ridge = RidgeRegresija()
    ridge.procjeni_koeficijente(X_train, Y_train)
    b = ridge.procijeni(X_test)

    # promjeniti odstupanje po potrebi
    odstupanje = 10500

    print("Koeficijeti ridge regresije su:(", ridge.b ,",", ridge.W[0],")")
    print("Pouzdanost ridge regresije sa odstupanjem", odstupanje, "je ",
          pouzdanost(b, Y_test, odstupanje) * 100, " posto")
    print("Ukupna greska ridge regresije je", ukupna_greska(b,Y_test))
    print("-------------------------------------------------------------------")

    # crtanje ridge regresije
    plt.scatter(X, Y, color="b", marker="s", s=30)
    plt.plot(X_test, b , color='r')
    plt.title('Ridge regresija')
    plt.xlabel('Godine iskustva')
    plt.ylabel('Plata')
    plt.show()


# postotak koliko gresaka je manje od nekog odstupanja
def pouzdanost(y_tacni, y_procijenjeni, odstupanje):
    pouzdanost = np.sum(abs(y_tacni - y_procijenjeni) < odstupanje) / len(y_tacni)
    return pouzdanost

# suma svih gresaka(apsolutna razlika y_procjenjeni i y_tacni)
def ukupna_greska(y_tacni , y_procijenjeni):
    greska = np.sum(abs(y_tacni - y_procijenjeni))
    return greska

def main():

    # u zavisnosti koji se dataset testira promijeniti labele x i y na crtezu i odstupanje

    # link sa kojeg je skinut ovaj dataset
    # https://www.kaggle.com/rohankayan/years-of-experience-and-salary-dataset
    # prva kolona predstavlja godine iskustva dok druga kolona predstavlja iznos plate za date godine iskustva

     df = pd.read_csv(r'C:\Users\Sanjin\Desktop\Plata_godine_iskustva.csv')
     X = df.iloc[:, :-1].values
     Y = df.iloc[:, 1].values
     x = df.iskustvo.values
     y = df.plata.values
     pokreniLinearnu(x,y)
     pokreniRidge(X,Y)


    # rucno dodani podaci
    # prva kolona predstavlja starost autama u godinama, a druga kolona cijenu auta za datu godinu u hiljadama

    #df2 = pd.read_csv(r'C:\Users\Sanjin\Desktop\Starost_cijena_auta.csv')
    #X = df2.iloc[:, :-1].values
    #Y = df2.iloc[:, 1].values
    #x = df2.starost.values
    #y = df2.cijena.values
    #pokreniLinearnu(x, y)
    #pokreniRidge(X, Y)



if __name__ == "__main__":
    main()

