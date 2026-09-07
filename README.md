# Cell Death — Simulation et validation de filtres particulaires

Ce dépôt contient des codes de simulation et de validation numérique pour un modèle spatial stochastique de mort cellulaire avec :

Le dépôt permet notamment de comparer plusieurs constructions de filtres particulaires en étudiant le biais de l'estimation d'une quantité intégrée liée à la zone dans laquelle une mort peut avoir lieu.

---

## 1. Modèle spatial

Le domaine spatial est

$$W=[0,L_x]\times[0,L_y].$$

Le modèle contient trois processus principaux :

* $V_t^a$ : processus latent des centres actifs de caspase ;
* $V_t^d$ : processus observé des morts cellulaires ;
* $V_t^p$ : processus des zones de protection ERK.

La zone active est définie par l'union des disques associés aux centres actifs :

$$A(V_t^a) = W\cap \bigcup_i B(Y_i^a,R_i^a).$$

La zone dans laquelle une mort peut effectivement se produire est

$$D_t = A(V_t^a)\setminus A(V_t^p).$$

---

## 2. Intensité d'activation

L'intensité spatiale d'activation dépend de la position $x$ et de l'état latent courant :

$$\lambda_a(x\mid V_{t-}^a) = \begin{cases} \lambda_{a,1}, & x\in A(V_{t-}^a),\\ \lambda_{a,T}, & x\notin A(V_{t-}^a) \text{ et }x\in T,\\ \lambda_{a,c}, & \text{sinon}. \end{cases}$$

où $T$ désigne une région spatiale fixe en forme de T.

Dans les simulations du dépôt, on impose

$$0\leq \lambda_{a,c} \leq \lambda_{a,T} \leq \lambda_{a,1},$$

ce qui permet d'utiliser un mécanisme de **thinning** à partir d'un processus dominant d'intensité $\lambda_{a,1}$.

---

## 3. Processus de mort cellulaire

Conditionnellement à l'état latent, l'intensité spatiale de mort est

$$\lambda_d(x,t) = \lambda_d \mathbf 1_{\{x\in A(V_{t-}^a)\}} \mathbf 1_{\{x\notin A(V_{t-}^p)\}}.$$

Une mort ne peut donc se produire que :

1. à l'intérieur d'une zone active ;
2. en dehors des zones de protection ERK.

Après chaque mort observée, une nouvelle zone de protection ERK est créée.

Les rayons sont simulés selon

$$R^a\sim \operatorname{Exp}(\beta_R^a), \qquad R^d\sim \operatorname{Exp}(\beta_R^d),$$

et les disparitions des zones actives et ERK sont gouvernées respectivement par les taux

$$\beta_T^a \qquad\text{et}\qquad \beta_T^d.$$

---

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
