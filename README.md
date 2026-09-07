# Cell Death — Simulation et validation de filtres particulaires

Ce dépôt contient des codes de simulation et de validation numérique pour un modèle spatial stochastique de mort cellulaire.

Le modèle prend en compte :

* l'activation latente de la caspase ;
* une rétroaction positive des zones actives ;
* les morts cellulaires observées ;
* la protection locale par ERK après une mort ;
* une hétérogénéité spatiale de l'intensité d'activation dans une zone fixe en forme de **T** ;
* l'approximation de l'état latent par plusieurs filtres particulaires.


# Structure du dépôt

```text
celldeath/
│
├── README.md
├── gillespiealgo .py
├── rejectionalgo.py
├── validate3pf.cpp
├── results.txt
├── plot.py
└── plot.png
```

Les différents fichiers correspondent à la simulation du modèle, à la validation des filtres particulaires et à la visualisation des résultats.

---

# 4. Simulation par algorithme de Gillespie

## `gillespiealgo .py`

Ce fichier implémente une simulation événementielle du modèle complet à l'aide d'un algorithme de type **Gillespie**.

Le programme considère quatre types d'événements :

1. proposition d'une activation ;
2. proposition d'une mort ;
3. disparition d'un centre actif ;
4. disparition d'une zone de protection ERK.

Si $N_t^a$ désigne le nombre de centres actifs présents au temps $t$ et $N_t^p$ le nombre de zones ERK actives, le taux total utilisé par l'algorithme est

$$
a_0
=
\lambda_{a,1}|W|
+
\lambda_d|W|
+
N_t^a\beta_T^a
+
N_t^p\beta_T^d.
$$

Le temps jusqu'au prochain événement est alors simulé selon

$$
\Delta t
\sim
\operatorname{Exp}(a_0).
$$

Le type du prochain événement est ensuite choisi proportionnellement aux quatre taux correspondants.

Le script fournit également une visualisation dynamique :

* des activations acceptées ;
* des centres actifs ;
* des zones actives ;
* des morts cellulaires ;
* des zones de protection ERK ;
* de la zone fixe en forme de T.

### Exécution

```bash
python "gillespiealgo .py"
```

---

# 5. Simulation par rejet

## `rejectionalgo.py`

Ce fichier fournit une implémentation alternative fondée sur la **simulation par rejet**, ou **thinning**.

Pour les activations, un processus ponctuel de Poisson dominant d'intensité $\lambda_{a,1}$ est d'abord simulé sur le domaine.

Chaque candidat situé en $x$ est ensuite accepté avec probabilité

$$
p_{\mathrm{acc}}(x)
=
\frac{\lambda_a(x\mid V_{t-}^a)}
{\lambda_{a,1}}.
$$

Par conséquent :

$$
p_{\mathrm{acc}}(x)
=
1
\qquad
\text{si }
x\in A(V_{t-}^a),
$$

$$
p_{\mathrm{acc}}(x)
=
\frac{\lambda_{a,T}}{\lambda_{a,1}}
\qquad
\text{si }
x\in T\setminus A(V_{t-}^a),
$$

et

$$
p_{\mathrm{acc}}(x)
=
\frac{\lambda_{a,c}}{\lambda_{a,1}}
$$

dans le reste du domaine.

Les candidats de mort sont conservés uniquement lorsqu'ils appartiennent à

$$
D_t
=
A(V_t^a)\setminus A(V_t^p).
$$

Le script fournit également une animation du système spatial au cours du temps.

### Exécution

```bash
python rejectionalgo.py
```

---

# 6. Validation des filtres particulaires

## `validate3pf.cpp`

Ce programme C++ simule des jeux de données selon le modèle événementiel, puis compare trois méthodes particulaires :

* `A3_local` ;
* `A4_full` ;
* `A5_empirical_optimal`.

Pour chaque intervalle compris entre deux morts observées successives, on considère la quantité

$$
B_k
=
\int_{S_{k-1}^d}^{S_k^d}
|D_t|\,dt.
$$

Ici, $|D_t|$ désigne l'aire de la région spatiale dans laquelle une mort peut avoir lieu au temps $t$.

Lors de la simulation des données, le programme calcule une valeur de référence

$$
B_k^{\mathrm{true}}.
$$

Les filtres particulaires produisent ensuite une approximation de la quantité conditionnelle associée à $B_k$ à partir des seules morts observées.

---

## 6.1. Filtre A3 — `A3_local`

Le premier filtre utilise un poids fondé essentiellement sur la compatibilité spatiale à l'instant de la mort.

Pour une trajectoire particulaire compatible avec la mort observée, le poids local utilisé dans le programme est

$$
G_k
=
\frac{
\mathbf{1}_{\{Y_k^d\in D_{S_k^d-}\}}
}{
|D_{S_k^d-}|
}.
$$

Si

$$
Y_k^d
\notin
D_{S_k^d-},
$$

alors

$$
G_k=0.
$$

Dans les résultats numériques, cette méthode est appelée

```text
A3_local
```

Cette construction utilise donc principalement l'information géométrique disponible à l'instant immédiatement antérieur à la mort observée.

---

## 6.2. Filtre A4 — `A4_full`

Le deuxième filtre utilise la vraisemblance complète de l'observation sur l'intervalle.

Son potentiel est

$$
G_k
=
\lambda_d
\exp(-\lambda_d B_k)
\mathbf{1}_{\{Y_k^d\in D_{S_k^d-}\}}.
$$

avec

$$
B_k
=
\int_{S_{k-1}^d}^{S_k^d}
|D_t|\,dt.
$$

Le terme

$$
\exp(-\lambda_d B_k)
$$

correspond au terme de survie sur l'intervalle

$$
(S_{k-1}^d,S_k^d).
$$

Plus explicitement,

$$
\exp(-\lambda_d B_k)
=
\exp
\left(
-\lambda_d
\int_{S_{k-1}^d}^{S_k^d}
|D_t|\,dt
\right).
$$

Dans les résultats numériques, cette méthode est appelée

```text
A4_full
```

Contrairement à `A3_local`, cette méthode tient donc compte de toute l'évolution de la zone de mort admissible entre deux observations successives.

---

# 6.3. Filtre A5 — `A5_empirical_optimal`

Le troisième filtre utilise une approximation empirique de la **proposition optimale**.

Pour chaque particule parent, le programme simule

$$
M_{\mathrm{prop}}
$$

segments latents candidats.

Pour le candidat $j$, le potentiel associé à l'observation est

$$
G_j
=
\lambda_d
\exp(-\lambda_d B_j)
\mathbf{1}_{\{Y_k^d\in D_{S_k^d-}^{(j)}\}}.
$$

On dispose donc de candidats

$$
H_k^{(1)},
H_k^{(2)},
\ldots,
H_k^{(M_{\mathrm{prop}})}
$$

avec leurs poids

$$
G_1,
G_2,
\ldots,
G_{M_{\mathrm{prop}}}.
$$

Le candidat finalement conservé est sélectionné avec une probabilité proportionnelle à son potentiel :

$$
\mathbb{P}
\left(
J=j
\mid
G_1,\ldots,G_{M_{\mathrm{prop}}}
\right)
=
\frac{G_j}
{\displaystyle\sum_{\ell=1}^{M_{\mathrm{prop}}}G_\ell}.
$$

Le programme utilise pour cela un **weighted reservoir sampling**.

Cette méthode permet d'effectuer cette sélection sans conserver simultanément en mémoire tous les candidats.

Le poids externe de la particule est estimé par

$$
\widehat{h}_k
=
\frac{1}{M_{\mathrm{prop}}}
\sum_{j=1}^{M_{\mathrm{prop}}}
G_j.
$$

Dans les résultats numériques, cette méthode est appelée

```text
A5_empirical_optimal
```

---

# 7. Approximation de l'intégrale spatiale

Le calcul exact de l'aire

$$
|D_t|
$$

peut être coûteux lorsque plusieurs disques actifs et plusieurs zones ERK se chevauchent.

Le programme utilise donc une grille de points dans $W$.

Si la grille contient

$$
m
=
\mathrm{GRID\_SIDE}^2
$$

points et si $n_D(t)$ désigne le nombre de points appartenant à $D_t$, alors

$$
|D_t|
\approx
n_D(t)
\frac{|W|}{m}.
$$

Cette approximation est utilisée pour calculer numériquement

$$
B_k
=
\int_{S_{k-1}^d}^{S_k^d}
|D_t|\,dt.
$$

---

# 8. Rééchantillonnage et ESS

Les filtres utilisent l'**Effective Sample Size** (ESS).

Pour des poids normalisés

$$
w_k^{(1)},\ldots,w_k^{(N)},
$$

l'ESS est définie par

$$
\operatorname{ESS}_k
=
\frac{1}{
\displaystyle
\sum_{i=1}^{N}
\left(w_k^{(i)}\right)^2
}.
$$

Dans le programme, un rééchantillonnage est effectué lorsque

$$
\operatorname{ESS}_k
\leq
0.75N.
$$

Après rééchantillonnage, les particules possèdent de nouveau des poids uniformes

$$
w_k^{(i)}
=
\frac{1}{N}.
$$

---

# 9. Estimation particulaire de $B_k$

Pour chaque segment $k$, le filtre produit une collection de valeurs

$$
B_k^{(1)},\ldots,B_k^{(N)}
$$

associées aux particules.

Après normalisation des poids, l'estimation particulaire utilisée est

$$
\widehat{B}_{k,N}
=
\sum_{i=1}^{N}
w_k^{(i)}
B_k^{(i)}.
$$

Cette quantité est ensuite comparée à la valeur simulée

$$
B_k^{\mathrm{true}}.
$$

---

# 10. Expérience de Monte-Carlo

Le programme considère plusieurs tailles de populations particulaires :

$$
N
\in
\{100,250,500,1000,2000,4000\}.
$$

Pour chaque valeur de $N$, l'expérience est répétée sur $R$ jeux de données simulés indépendamment.

Les valeurs par défaut sont :

```text
R = 500
K = 20
GRID_SIDE = 40
M_PROP = 20
ESS threshold = 0.75
```

où :

* `R` est le nombre de répétitions Monte-Carlo ;
* `K` est le nombre de morts observées utilisées dans chaque jeu de données ;
* `GRID_SIDE` contrôle la résolution de l'approximation spatiale ;
* `M_PROP` est le nombre de candidats utilisés par `A5_empirical_optimal`.

---

# 11. Compilation de `validate3pf.cpp`

Le programme nécessite un compilateur compatible avec **C++17**.

Sous Linux, avec OpenMP :

```bash
g++ -std=c++17 -O3 -DNDEBUG -fopenmp validate3pf.cpp -o validate3pf
```

Puis :

```bash
./validate3pf
```

Les arguments facultatifs sont

```text
./validate3pf R GRID_SIDE M_PROP
```

Par exemple :

```bash
./validate3pf 500 40 20
```

Pour enregistrer directement les résultats :

```bash
./validate3pf 500 40 20 > results.txt
```

---

# 12. Format des résultats

Le programme produit d'abord une ligne décrivant la configuration, par exemple :

```text
R=500  K=20  area_points=1600  ESS_threshold=0.75  M_prop=20
```

puis un tableau dont l'en-tête est

```text
N k algorithm success collapse mean_true_paired mean_PF bias abs_bias paired_MCSE
```

Les colonnes ont la signification suivante :

| Colonne            | Signification                                           |
| ------------------ | ------------------------------------------------------- |
| `N`                | nombre de particules                                    |
| `k`                | numéro du segment                                       |
| `algorithm`        | méthode particulaire utilisée                           |
| `success`          | nombre de répétitions réussies                          |
| `collapse`         | nombre d'effondrements du système de particules         |
| `mean_true_paired` | moyenne des valeurs vraies sur les répétitions réussies |
| `mean_PF`          | moyenne des estimations particulaires                   |
| `bias`             | biais moyen                                             |
| `abs_bias`         | valeur absolue du biais                                 |
| `paired_MCSE`      | erreur standard Monte-Carlo du biais apparié            |

Pour une répétition $r$, on définit l'erreur

$$
E_{r,k,N}
=
\widehat{B}_{r,k,N}
-
B_{r,k}^{\mathrm{true}}.
$$

Le biais empirique est alors

$$
\operatorname{Bias}_{k,N}
=
\frac{1}{R_{\mathrm{succ}}}
\sum_{r=1}^{R_{\mathrm{succ}}}
E_{r,k,N},
$$

où $R_{\mathrm{succ}}$ désigne le nombre de filtres n'ayant pas subi d'effondrement.

La quantité `paired_MCSE` mesure l'erreur Monte-Carlo associée à cette estimation du biais.

---

# 13. Visualisation des résultats

## `plot.py`

Ce script lit le fichier

```text
results.txt
```

et compare les trois filtres :

```text
A3_local
A4_full
A5_empirical_optimal
```

Pour chaque segment $k$, le graphique représente le biais

$$
\operatorname{Bias}_{k,N}
$$

en fonction du nombre de particules $N$.

Les barres d'erreur correspondent à

$$
\operatorname{paired\_MCSE}_{k,N}.
$$

L'axe horizontal est affiché en échelle logarithmique afin de faciliter la comparaison entre les différentes valeurs de $N$.

---

## Dépendances Python

Les scripts Python utilisent principalement :

```text
numpy
matplotlib
pandas
```

Installation :

```bash
pip install numpy matplotlib pandas
```

---

## Exécution de `plot.py`

Pour utiliser les noms de fichiers par défaut :

```bash
python plot.py
```

Le programme lit alors

```text
results.txt
```

et produit

```text
plot.png
```

Il est également possible de préciser explicitement les fichiers :

```bash
python plot.py results.txt plot.png
```

---

# 14. Résultat graphique

La figure produite par le programme est affichée directement dans ce README :

![Comparaison du biais des trois filtres particulaires](plot.png)

Cette figure permet de comparer, pour chaque segment $k$, le comportement des trois méthodes lorsque le nombre de particules augmente.

L'objectif est notamment d'étudier si le biais se rapproche de zéro lorsque

$$
N\longrightarrow\infty.
$$

---

# 15. Workflow recommandé

## Étape 1 — Simuler le modèle

```bash
python "gillespiealgo .py"
```

ou utiliser la version fondée sur le thinning :

```bash
python rejectionalgo.py
```

## Étape 2 — Compiler le programme de validation

```bash
g++ -std=c++17 -O3 -DNDEBUG -fopenmp validate3pf.cpp -o validate3pf
```

## Étape 3 — Lancer l'expérience Monte-Carlo

```bash
./validate3pf 500 40 20 > results.txt
```

## Étape 4 — Générer la figure

```bash
python plot.py results.txt plot.png
```

La chaîne complète est donc

```text
simulation du modèle
        ↓
observations de morts cellulaires
        ↓
A3 / A4 / A5
        ↓
estimation de B_k
        ↓
comparaison avec B_k^{true}
        ↓
biais + MCSE
        ↓
plot.png
```

---

# 16. Résumé des trois filtres

| Méthode                | Principe                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `A3_local`             | Poids fondé sur l'état spatial au moment de la mort                                   |
| `A4_full`              | Poids utilisant la vraisemblance complète avec le terme de survie                     |
| `A5_empirical_optimal` | Approximation empirique de la proposition optimale avec $M_{\mathrm{prop}}$ candidats |

Le dépôt permet ainsi d'étudier expérimentalement l'effet du choix du potentiel et de la proposition particulaire sur la qualité de reconstruction d'une quantité latente du processus de mort cellulaire.

---

# 17. Auteur

**Junwen Xiao**

Projet consacré à la simulation de modèles ponctuels spatiaux de mort cellulaire et à la validation numérique de méthodes de filtrage particulaire.
