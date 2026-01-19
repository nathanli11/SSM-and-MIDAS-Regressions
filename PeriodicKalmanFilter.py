import numpy as np
from numpy.linalg import inv, norm

class PeriodicKalmanFilter:
    """
    Filtre de Kalman periodique SSM
    """

    def __init__(self, G, Q, Z_list:list, tol=1e-12, max_iter=20000):
        """
        G: matrice de transition (n x n)
        Q: matrice de covariance des innovations etat (n x n)
        Z_list: liste [Z1, ..., Zm] des matrices d'observation périodiques
        tol : tolérance numérique pour la convergence du Riccati periodique
        max_iter : nombre maximum d'itération du Riccati
        """

        self.G = G
        self.Q = Q
        self.Z_lsit = Z_list
        self.m = len(Z_list)
        self.n = G.shape[0]
        self.tol = tol
        self.max_iter = max_iter

        self.P_list = None
        self.K_list = None
        self.A_list = None

    def solve_steady_state(self):
        """
        Résoud les equations de Riccati périodique (Eq. 2.8)
        Calcule P_j|j-1, K_j|j-1 et A_j|j-1
        """

        # Initialisation (matrice identite)
        P = [np.eye(self.n) for _ in range(self.m)]

        for _ in range(self.max_iter):
            # Sauvegarde pour tester la convergence
            P_old = [p.copy() for p in P]

            for j in range(self.m):
                # Matrice d observation à la phase j
                Z = self.Z_list[j]
                Pj = P[j]

                # Eq 2.8 : variance de l'innovation d'observation
                S = Z @ Pj @ Z.T

                # Equation 2.8 (avec inv(S))
                P_next = (
                    self.Q 
                    + self.G @ Pj @ self.G.T
                    - self.G @ Pj @ Z.T @ inv(S) @ Z @ Pj @ self.G.T
                )

                # Equation 2.8 (sans inv(S))
                tmp = np.linalg.solve(S, Z @ Pj @ self.G.T)
                P_next = (
                    self.Q 
                    + self.G @ Pj @ self.G.T
                    - self.G @ Pj @ Z.T @ tmp
                )

                P[(j+1) % self.m] = P_next
            
            # Test de convergence
            err = max(norm(P[j] - P_old[j]) for j in range(self.m))
            if err < self.tol:
                break
        else:
            raise RuntimeError("Riccati périodique non convergent")
        
        # Calcul des gains et matrices A (Eq. 2.9)
        self.P_list = P
        self.K_list = []
        self.A_list = []

        for j in range(self.m):
            Z = self.Z_list[j]
            Pj = P[j]
            S = Z @ Pj @ Z.T
            
            # Gain de Kalman
            # K = Pj @ Z.T @ inv(S)
            K = np.linalg.solve(S.T, (Z @ Pj).T).T

            # Matrice de transition (Eq 2.10)
            A = self.G - K @ Z @ self.G

            self.K_list.append(K)
            self.A_list.append(A)

        return self
    
    def filter_state(self, Y_seq):
        """
        Applique le filtre de Kalman à gains fixes (Eq. 2.10)

        Y_seq[t] : vecteur d'observation à la date t dont la dimension
        dépend de la phase (x seul ou (y,x))
        """

        # Initialisation du filtre
        a = np.zeros(self.n)

        for t, y in enumerate(Y_seq):
            j = t % self.m
            K = self.K_list[j]
            Z = self.Z_list[j]
            A = self.A_list[j]

            # Eq 2.10 de alpha chapeau
            a = A @ a + K @ y
        return a
    
    def forecast_from_state(self, a_t, H, h):
        """
        Calcule la prévision h trismestres ahead
        Calcule E[y_{t+h} | I_t^M]

        H est le vecteur d'extraction de y
            ex: [gamma1, 1, 0] car y* = gamma1 f + u1
        """
        # Propagation de l'état de mh pas haute fréquence
        Gmh = np.linalg.matrix_power(self.G, self.m * h)

        # Correspond a y_hat (t+h sachant t) = HG^(mh)a_hat
        return float(H @ (Gmh @ a_t))
    
    def kalman_weights(self, H, h, Kbar):
        """ 
        Calcule les poids du filtre de Kalman par méthode d'impulsion
        Implémentation de 3.3
        """

        N = self.m * (Kbar + 1)

        def zero_sequence():
            # Crée une sequence de zero
            seq = []
            for t in range(N):
                phase = t % self.m
                p = self.Z_lsit[phase].shape[0]
                seq.append(np.zeros(p))
            return seq
        
        # poids sur y(t-j)
        w_y = np.zeros(Kbar + 1)

        for j in range(Kbar + 1):
            seq = zero_sequence()
            idx = N - 1 - j * self.m
            seq[idx][0] = 1.0

            # Injection d'une impulsion de 1 sur y(t-j)
            a = self.filter_state(seq)
            w_y[j] = self.forecast_from_state(a, H, h)
        
        w_x = np.zeros(self.m * Kbar + 1)
        for k in range(self.m * Kbar + 1):
            seq = zero_sequence()
            idx = N - 1 - k
            phase = idx % self.m

            # On observe que x avant la fin du trimestre
            if self.Z_lsit[phase].shape[0] == 1:
                seq[idx][0] = 1.0
            # Fin du trimestre, on observe (y,x)
            else:
                seq[idx][1] = 1.0
            a = self.filter_state(seq)
            w_x[k] = self.forecast_from_state(a, H, h)

        return w_y, w_x