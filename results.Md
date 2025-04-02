# Data generation 

For the analysis, I generated synthetic data to simulate how both treatment assignment and outcomes spread through a network, reflecting realistic patterns of social interaction and structural dependencies. This simulation framework allows to explicitly control the data-generating process, including the influence of covariates, network structure, and treatment assignment. 

By varying these components and applying different estimation strategies, it can be evaluated how network interference influence accuracy of estimations and are network adjastment strategies effective.

All simulation parameters are summarized in Table 1.

| Parameter              | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| **Network Models**     | Barabasi–Albert, Watts–Strogatz, Erdos–Renyi; SBM with covariate-based blocks |
|**Observations**| 10.000 nodes|
| **Neighbor Influence** | Small (0.1), Medium (0.3), Large (0.7)        |
| **Covariates**         | 2 features per node; each sampled from $N(\mu, \sigma)$, with $\mu$ ∈ [3, 10] and σ ∈ [0.4, 0.8]             |
| **Treatment Effect**   | Medium (0.3)               |
|**Group share**|  |
| **Treatment Assignment** | Randomized or observational (based on individual and neighbor covariates)          |
| **Outcome Model**      | Spatial Autoregressive Model |

**Table 1**: Key parameters and components used in the data generation and estimation process.


## Treatment assignment 

The primary focus of this study is on randomized treatment assignment, which is commonly used by researchers to insure well-balanced treated and control groups and facilitates unbiased estimation of treatment effects. In the baseline setup, half of the nodes are randomly assigned to treatment, independent of covariates or network position. This setting allows us to evaluate how different network-aware adjustment methods perform under experiment conditions.

To assess the applicability of these methods in observational settings, I also simulate scenarios where treatment assignment depends on both individual covariates and its neigbours. In particular, this mechanism involves combining each node's own covariates with the aggregated covariates of its neighbors, capturing social influence. A linear score is computed from this combined feature set, normalized to a range from zero to one, and treatment is assigned to a proportion of nodes with the lowest scores. 

For both scenarios, groups are split equally and a medium treatment effect of 30% is used. The effect is implemented additively: treated nodes receive an outcome shift drawn from a normal distribution with mean 0.3 and variance 0. This ensures that the treatment effect is consistent across treated units, simplifying both the interpretation of results and the evaluation of bias in different estimation methods.


## Network generation 
Different types of networks are used to account for various structural patterns observed in real-world networks. This diversity allows to model a range of scenarios, such as friendship groups, online social platforms, or entirely random interactions. Since it remains unclear whether and how the type of network structure affects the estimation bias under network interference, exploring multiple models is essential. For each network type, a graph with ten thousand nodes is generated. These networks are represented by an adjacency matrix, denoted W, which encodes the presence or absence of edges between nodes and serves as the foundation for subsequent simulations and analysis.

### Random network model
The random network models considered in this study — Barabasi–Albert, Watts–Strogatz, and Erdos–Renyi. In these models, the network structure is formed independently of any node attributes, making them suitable for isolating the role of structure and other network features in estimation bias, as they reflect different structural assumptions:

The *Barabasi-Albert model* is well suited to simulating real-world systems that exhibit scale-free properties, such as friendship groups, high school social networks, or online platforms, where a small number of nodes tend to accumulate a disproportionately large number of connections \cite{bertotti2019configuration}. 

The *Watts-Strogatz model* is generated to create small-world networks, characterized by high clustering and short average path lengths, features commonly observed in many social and biological systems. This model is useful for studying phenomena like information diffusion or community formation, as it captures the balance between local and global connectivity within a network \cite{song2014simple}.

The *Erdos-Renyi model* generates random networks in which each possible edge between nodes has a fixed probability. This model is often used as a benchmark to understand how randomness alone can shape network structures and dynamics. While it lacks the clustering seen in the Watts-Strogatz model or the hubs in the Barabási-Albert model, it provides a baseline for comparison by modelling purely stochastic connections.

### SBM
In real-world networks, the structure of connections between nodes is often closely related to their observable characteristics. For example, people with similar features, such as age, gender, interests tend to form connections more frequently. It leads to covariates and network structure not beeing independent. This phenomenon is commonly reffered as homophily. 

To model this dependency explicitly, Stochastic Block Model (SBM) with covariate-driven block assignments is used. SBM allows to generate networks where the probability of a connection between two nodes depends on the communities they belong to, and these communities are determined directly from covariates. This results in more realistic simulation of social networks. 

Nodes are grouped into blocks based on their covariates, using k-means clustering. The probability of an edge existing between any two nodes depends on their assigned blocks, as defined by an inter-block probability matrix. Three blocks are generated, edges within first block are connected with probability of 0.8, second - 0.7 and third - 0.6, nodes from different blocks are connected with probability of 0.1. 

This methods also provides a framework for simulating observational studies where treatment assignment and outcomes may be influenced by both individual covariates and network structure.

### Corrupt network 
Sometimes the true network structure is unknown, because networks are often inferred from noisy observations or inclomplete data. Researchers can also use different type of data to establish connections that is only approximate to the true one. In many practical situations researchers deal with proxy of real network, which raises a question of robustness of adjustment methods in these settings. 

To investigate this, I introduced noise into the Barabasi-Albert graph by randomly adding edges with probability 0.2 and removing them with probability 0.3, simulating a realistic scenario where the observed graph deviates from the true one.

### Covariates 
Covariates represent individual-level characteristics, such as age, income, or behavioral traits, and are used to simulate observed heterogeneity across nodes. In this setup, two covariates are generated for each individual. Each covariate is sampled from a normal distribution with a randomly chosen mean in the range [3, 10] and standard deviation in the range [0.4, 0.8]. These distributions are not grounded in any theory but are chosen to introduce more realistic variability, which can be benificial for simulations. This setup allows for diversity in individual features, which may influence both treatment assignment and outcomes in observational settings.

### Outcome

To simulate outcomes in the presence of network interference, I use the Spatial Autoregressive (SAR) model — a widely adopted approach for modeling dependencies between connected units. The SAR model explicitly accounts for the fact that an individual’s outcome may be influenced not only by their own characteristics but also by the outcomes of their neighbors in the network. This setup reflects real-world scenarios where behaviors or responses tend to cluster or diffuse along social ties.

A key assumption of the SAR model is indirect interference: spillover effects from neighbors are proportional to the strength of the direct effect. In other words, if the treatment has no direct effect on a unit, here will be no spillovers to its neighbours. This assumption is integral to the linearity of the model and serves as a foundation for simulating outcomes in networks.

Formally, the outcome can be expressed as: $y = (I - \rho W)^{-1} X\beta + (I - \rho W)^{-1} \epsilon$, where $\rho$ represents the influence of neighbors, $W$ is the adjacency matrix capturing network connections, $X$ is a matrix of covariates, and $\epsilon$ represents independent noise. 

The coefficients $\beta$ are sampled from a normal distribution $N(3, 10)$, and the neighbour influence parameter $\rho$ will take on values corresponding to small (0.1), medium (0.3), and large (0.7) levels of spillover intensity. This allows to simulate different scenarios, enabling an understanding of how outcome dependencies affect estimation bias and the performance of network adjustment methods.

# Methods

## Estimand and estimation strategies
In this study, the estimand of interest is the Average Treatment Effect (ATE), defined as the average difference in potential outcomes between the treated and control groups. The ATE provides a measure of the impact of the treatment assignment in the population when SUTVA holds.

### Linear regression 
In experimentational settings, as treatment is assigned randomly, to estimate effect I used the following linear regression: $y = \beta_0 + \beta_1*Covariate_{i} + \beta_2*Group_{i} + \epsilon_{i}$, where $Group_i$ indicates individuals' group and $Covariate_{i}$ individual covariates. The key expectation without interference is that the estimated treatment effect, represented by $\beta_2$ should align with the true effect used to generate the data when there is no bias. 

## Prospenity matching 
In observational settings, treatment assignment is not randomized and can depend on observed and potentially unobserved covariates. This introduces confounding, which can bias estimation of the ATE. To address this, I apply Propensity Score Matching (PSM) — a widely used technique in these settings. 

Its core idea is to estimate the propensity score, whish is a probability of receiving treatment given the observed covariates. In this study, a logistic regression model is used for this. These scores are then used to match treated units with similar control units, under the assumption that treatment assignment is conditionally independent of outcomes, given the propensity score.

To ensure quality matches, the matching is performed with a caliper (fixed to 0.05) — a maximum allowed distance in propensity scores between matched units — and a fixed number of nearest neighbors (fixed to 5). For each treated unit, the algorithm identifies the closest control unit within the caliper threshold based on the Euclidean distance in propensity scores. If a suitable match is found, it is included in the matched sample.

This matching procedure reduces covariate imbalance between the treatment and control groups, producing a sample that  approximates the conditions of an experiment. The treatment effect is then estimated using linear regression, which was discussed previously. 

## Network adjustments 

To account for the bias that arises from network interference, network information can be incorporated into estimation methods. In this way, the estimation can become more robust to network interference, providing more accurate estimates of treatment effect. One approach to account for network interference is to include network-aware covariates in the estimation method, such as linear regression. These covariates aim to adjust for structural characteristics that may confound the treatment effect.

First, **network centrality** measures can be used, including:
    - *Degree centrality* — the number of direct connections a node has, representing its level of exposure or influence. 

    - *Betweenness centrality* — how often a node lies on the shortest paths between other nodes, capturing its role as a bridge within the network.

Second, **community membership** can be incorporated, identifying the subgroups or clusters to which nodes belong. This reflects the fact that individuals in the same group often have similar treatment or outcomes.

Thirdly, more complex node representations can be used. **Node embeddings** from Node2Vec provide a way to represent each node in the network as a dense vector of numbers that captures structural properties of its position in the graph. This is a graph embedding algorithm that generates vector representations of nodes by simulating random walks through the network. These walks create sequences of nodes that reflect the structure of the graph — either staying close to the starting node (capturing local neighborhoods) or exploring more distant connections (capturing broader structure). These sequences are then treated similarly to sentences in natural language, and embeddings are learned using methods inspired by Word2Vec, where nodes that appear in similar contexts receive similar vector representations.

The result is a low-dimensional embedding for each node, where structurally similar nodes tend to have similar vectors. Unlike traditional hand-crafted features like centrality, Node2Vec embeddings can capture both local context (who the neighbors are) and global position (how the node is embedded in the broader network structure). This makes them especially useful in settings with complex or noisy graphs, as they can encode subtle relational patterns that simpler features may overlook.

Beyond this it's can be crucial to account for the impact of treatment through the network. The **count of treated neighbors** directly reflects the level of peer exposure by summing the number of adjacent nodes that have received treatment. This metric models local spillover effects and partially controls for outcome changes driven by neighbor influence. The **treatment exposure map** offers a more refined measure by assessing the fraction or weighted proportion of treated neighbors. Unlike simple counts, this metric captures the intensity of treatment exposure from the surrounding network, allowing for a more precise evaluation of neighbors' influence on individual outcomes.


# Results
This section outlies results of simulations.

## Experimentational design

Naive estimations that don't account for the presence of network interference produce consitently high bias of effect estimations across different networks and levels of neighbor influence, with an average bias of over 100%. Specifically, the bias reaches 126% for Barabasi–Albert, 100% for Watts–Strogatz, and 96% for Erdos–Renyi networks. As expected, the bias increases with the level of neighbor influence, confirming that stronger spillover effects amplify the bias in the effect estimation when interference is not accounted for.

To mitigate this bias, several network-aware adjustment strategies were tested. Incorporating centrality-based features (such as degree and betweenness centrality, optionally combined with community membership) into the regression significantly reduces the bias. In Barabasi–Albert networks, this adjustment brought the bias down to as low as 7.5–18% at moderate influence levels. However, at the high level of influence, the effectiveness declined, with some cases still resulting in biases of 40–45%. A similar pattern is observed for other types of networks, where centrality-based adjustment typically decrease the bias to 12% and 40%, depending on the feature set and influence level.

Including only community membership was less reliable. While it slightly improved bias in case of Watts–Strogatz and Erdos–Renyi networks structure, in other case it led to biases exceeding 100%. This suggests that while community structure captures some clustering effects, it is insufficient on its own to fully adjust for interference.

Adjustments using Node2Vec embeddings showed more robust performance across all network types. Bias typically remained in the range of 29–46%, regardless of the network type or influence level. While not fully eliminating bias, including them provided more stable results.

Both the count of treated neighbors and the treatment exposure map showed clear improvements. The treatment exposure feature performed better overallm it reduced the bias to as low as 3.7% in Erdos–Renyi, 9% in Barabasi–Albert, and 22–29% in Watts–Strogatz networks. In contrast, the raw count of treated neighbors was more sensitive to node degree and performed worse in networks with heterogeneous degree distributions, where the bias remained above 90% in high-influence cases.

Taken together, these results suggest that no single adjustment strategy is universally optimal, but some are clearly more robust. Treatment exposure maps and node embeddings consistently perform well across network types and influence levels. In contrast, community-only features or raw counts tend to be unstable or insufficient. Combining network structure and treatment spillover information appears essential for effectively reducing bias due to interference, especially in settings with high connectivity or strong peer effects.


| Network Type         | Adjustment Method                      | Avg. Bias (%) | ρ = 0.3 (%) | ρ = 0.6 (%) | ρ = 0.9 (%) |
|----------------------|----------------------------------------|---------------|-------------|-------------|-------------|
| **Barabási–Albert**  | Naive                                  | 126.0         | ~98.3       | ~137.9      | ~143.7      |
|                      | Degree + Betweenness                   | 20.7          | 20.7        | 7.5         | 42.5        |
|                      | Degree + Betweenness + Community       | 16.0          | 19.9        | 9.4         | 43.0        |
|                      | Community only                         | 143.7         | 95.5        | 137.9       | 143.7       |
|                      | Node2Vec Embeddings                    | 42.0–45.0     | 44.6        | 45.0        | 38.7        |
|                      | Treated Neighbors (Count)              | 89.3          | 33.0        | 96.3        | 138.6       |
|                      | Treatment Exposure                     | 45.9          | 9.0         | 90.3        | 85.8        |
| **Watts–Strogatz**   | Naive                                  | 100.0         | 95.8        | 122.3       | 73.6        |
|                      | Degree + Betweenness                   | 11.3          | 11.3        | 3.0         | 2.1         |
|                      | Degree + Betweenness + Community       | ~5.0          | 35.4        | 29.1        | 2.1         |
|                      | Community only                         | 73.5          | 35.4        | 122.3       | 73.5        |
|                      | Node2Vec Embeddings                    | 40.0–45.0     | 35.7        | 50.0        | 29.8        |
|                      | Treated Neighbors (Count)              | 40.6          | 44.4        | 29.1        | 48.3        |
|                      | Treatment Exposure                     | 29.8          | 22.5        | 22.5        | 28.8        |
| **Erdős–Rényi**      | Naive                                  | 96.0          | 60.0        | 91.0        | 134.9       |
|                      | Degree + Betweenness                   | 28.2          | 28.2        | 36.9        | 29.2        |
|                      | Degree + Betweenness + Community       | ~27.0         | 28.2        | 35.8        | 29.4        |
|                      | Community only                         | 135.9         | 60.6        | 88.6        | 134.9       |
|                      | Node2Vec Embeddings                    | 36.9          | 27.7        | 36.9        | 48.9        |
|                      | Treated Neighbors (Count)              | 29.1          | 25.6        | 29.1        | 32.7        |
|                      | Treatment Exposure                     | 22.0          | 3.7         | 27.7        | 21.3        |
**Table 2**: Percentage bias in treatment effect estimates across network types and adjustment methods.


### Corrupt network
The results show that naive estimations under the corrupted network remain highly biased. At a moderate neighbor influence level of 0.6, it results in a bias of 49.4%, while under high influence (0.9), the bias rises dramatically to 163.6%. This is consistent with what is observed under the true Barabasi-Albert graph, but the magnitude of bias under the corrupted graph is notably higher — particularly at strong influence levels. That means that corruption increases the estimation error.

When adjusting for network features such as degree centrality and betweenness centrality, the model achieves a bias reduction to ~39–45%, depending on the level of influence. Although this is a meaningful improvement, it is less effective than under the true network, where the same feature set brought the bias down to 16%. This highlights the sensitivity of centrality-based adjustments to structural distortions in the network.

The use of community features proves particularly fragile under network corruption. While under the true graph, community-based adjustment sometimes improved estimates, under the corrupted Barabasi-Albert graph it performs inconsistently and, in some cases, fails and leads to bias levels that reachs 163.6% under high influence. 

In contrast, Node2Vec embeddings show relatively better robustness, it reduces bias to 35–46%, only modestly worse than the 42–46% achieved using the true graph. This suggests that embeddings preserve enough structural information to partially compensate for missing or noisy edges.

In summary, while adjustment methods improve treatment effect estimation even under corrupted networks, their performance degrades compared to when the true network is available. Node embeddings appear more robust to  noise, as they are able to preserve enough structural information to partially compensate for incomplete data. Compared to that, community detection is especially vulnerable to network noise, potentially generating misleading results. Such network features, as node centrality, also produces not robust results but still can be used in practice.

## Observational study 

In the observational setting, where treatment assignment is based on both individual covariates and aggregated neighborhood characteristics, adjustment strategies proved largely ineffective in correcting for network-induced bias. Across all tested adjustments—ranging from raw covariates to degree centrality, community membership, treated neighbor counts, and node embeddings—the observed bias in estimated treatment effects remained consistently high. Under low and medium levels of neighbor influence, percentage bias remained in the range of 89–108%, regardless of the feature set used. Even advanced representations such as Node2Vec embeddings or exposure-based adjustments failed to meaningfully reduce estimation error. At high influence levels, the situation deteriorated further, with bias reaching up to 128%. These findings suggest that in observational network settings—where both treatment and outcome processes are shaped by latent community structure and peer effects—standard covariate and network-feature adjustments are insufficient. Correcting for confounding in such settings likely requires modeling approaches that explicitly account for interference and the joint distribution of treatment, covariates, and network structure.
