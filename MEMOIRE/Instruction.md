JE dois faire un mémoire mais de présentation de 15 min. 

Enfaite je voudrai pouvoir faire le meilleure mémoire et le plus pousser possible. 
En gros l'idée est de montrer que les optimisations de maintenant enfin les plus récentes qui sont boosté à l'ia , le machine learning etc sont plus efficaces que des optimisations dites basiques comme black litterman , max div , markowitz etc et j'en passe. Et surtout quel perspectives le calcul quantique à la dedans ( dans cette partie je vais m'aider à coder et à comprendre grace à certain papier de recherche. )

Donc pour l'instant on à fait markowitz , risk parity , black litterman etc. 
J'ai voulu décomposer en petite partie donc une partie pour voir ce que ca faisais tout seul etc mais je ne sais pas si c'est vraiment l'idéale. 

La l'idée c'est qu'on fasse une super strucutre , je veux que l'on fasse maintenant un modèle avec xgboost mais je voudrai surtout que l'on fasse un algorithme dans le genre :

Dans un premier temps il faudra que l'on charge l'univers et qu'on mette proprement le dataframe. 
Ensuite grace à un papier de recherche enfin plusieurs qui sont ici et que je vais te laisser lire : "C:\Users\TONY B\OneDrive\documentation\finance pur\portfolio optimization model for stock prediction with machine learning.pdf"; "C:\Users\TONY B\OneDrive\documentation\finance pur\Machine Learning for Quantitative Finance.pdf"; "C:\Users\TONY B\OneDrive\documentation\finance pur\Quantitative finace applications for machine learning experts.pdf"

MAis le plus important c'est le premier parce que en gros l'idée est de pouvoir prédire leur rendement sur une fenetre d'entrainement de 14 ans et de prédire sur 1 ans ca nous laisse de la marge et je pense que c'est bien, je pense qu'il faudra normaliser également pour éviter que les données soit trop tirés vers des valeurs extrêmes.  En gros l'idée est de pouvoir les ranker par rapport à leurs prédiction du meilleur au pire. et ca nous créera un vecteur d'actif avec Aa_1 , Ab_2 , Ac_3, ... , An_k. 
Donc la il faut qu'on est toute les métriques utiles , genre R² , intercept etc , pvalue...
Ensuite je voudrai donc passer à l'optimization de portefeuille.
L'idée ici est de pouvoir Faire une maxdiversification avec skfolio si c'est possible. 
Donc enfaite je veux pouvoir avoir mon vecteur de ranking de rendement en tant que matrice dasn cette formule : DR(w)=w⊤Σw​w⊤σ​
je ne sais pas si c'est possible ou sil il faut faire une interpolation linéaire ou si il faut multiplier par un autre vecteur mais en gros l'idée est de pouvoir avoir ce vecteur de ranking de rendement. 

Concrètement I want to implement a custom Max Diversification portfolio optimization using skfolio. The standard Max Diversification maximizes the Diversification Ratio:
DR(w)=w⊤σw⊤ΣwDR(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}DR(w)=w⊤Σw​w⊤σ​
Instead, I want to modify the numerator by replacing σ\sigma
σ with σ⊙μ\sigma \odot \mu
σ⊙μ, where μ\mu
μ is a vector of predicted future returns from an XGBoost model (normalized and winsorized), and ⊙\odot
⊙ is the element-wise product. The new objective becomes:
max⁡ww⊤(σ⊙μ)w⊤Σws.c.1⊤w=1, w≥0\max_w \frac{w^\top (\sigma \odot \mu)}{\sqrt{w^\top \Sigma w}} \quad \text{s.c.} \quad \mathbf{1}^\top w = 1,\ w \geq 0wmax​w⊤Σw​w⊤(σ⊙μ)​s.c.1⊤w=1, w≥0
The idea is to maximize diversification while tilting towards assets with the highest predicted alpha. Please implement this in skfolio, with μ\mu
μ as an external input vector. Show how to plug in a custom objective or override the standard MDP formulation

Et je veux y poser des contraintes  : 

En gros je veux maximiser l'alpha également donc on devra tirer dans les deux sens mais on va s'éaider de la frontière de pareto ou markowitz. je veux que les poids soit égale à un et si c'est possible je voudrai que les poids soit au min = à 10 bp. 

Donc voila l'idée de l'optimisation de portefeuille.




Par ailleurs une fois fini je voudrai par conséquent que tu me code un backtest on peut voir ensemble comment le configurer. 