### DGP
I assume relational network effect, meaning that the outcome of te unit depends on its covariates, as well as their neighbours' outcome and covariates. There also could be a positional network effect, which depicts an effect on a network level, meaning that unit's position in network affects outcome and structural effect that depicts effect of the network as a whole. For simplicity I don't assume this. 

Additionally, we can simulate whether the influence of every node is the same by normalizing the adjacency matrix. Hovewer, in real-world scenarios I find this assumption too simplistic. 

To simulate these scenarios we can apply spatial lag model approaches. The value of the outcome of one unit depends on the value of its neighbours, leading to dependencies between neighbours. This means that a system of equations has to be solved simultaneously: $$ y = X\beta + \rho Wy + \epsilon $$

Solving for y:

$$y = (I - \rho W)^{-1}X\beta + (I - \rho W)^{-1}\epsilon$$

Then, we can simulate treatment $Z$, which results in the following model: $$y = (I - \rho W)^{-1}X\beta_{1} + (I - \rho W)^{-1} (\beta_{2}*Z) + (I - \rho W)^{-1}\epsilon$$

## Prelimenary results 

#### Random network structure 
I generated several networks with 100 nodes: Barabasi-Albert, Watts-Strogatz,  Erdos-Renyi. For adjusting for the network effects I included the following features as covariates in a linear regression: 
    - network fearures: degree centrality, betweenness centrality, community;
    - node embedding from Node2Vec;
    - count of treated neighbours;
    - treatment exposure map: the extent to which a node in a network is influenced by treated neighbors. 


Naiv estimations that don't account for the presence of network effect have on average 107% bias. Different network significantly impact the bias, with Erdos-Renyi 96%, Barabasi-Albert 126% and Watts-Strogatz 100%. With the increase of neigbour influence the bias is also increasing. 

The inclusion of network properties such as degree centrality, betweenness centrality, and community structure appears to influence the bias, leading to 16% bias for Barabasi-Albert graph and 5% for other graph models. On the other hand, inclusion of just community feature decreased only bias in Watts-Strogatz model and for other network type even increased the bias. Adjustments using vector representation of nodes that were obtained using node2vec also decreases bias to around 42-46% for every network type. Using count of treated neighbours decreased the bias to 90% for Barabasi-Albert, 30% for Erdos Renyi, and 40% for Watts Strogatz. Using treatment exposure also decreases the bias to 45%, 22% and 29% respectively. 

Using count of treated neighbours 

(add table here)

#### Network structure dependant on covariates
The Stochastic Block Model is used to create random graphs with a community structure. Nodes are grouped into blocks based on their covariates, using k-means clustering. The probability of an edge existing between any two nodes depends on their assigned blocks, as defined by an inter-block probability matrix. Three blocks are generated, edges within first block are connected with probability of 0.8, second - 0.7 and third - 0.6, nodes from different blocks are connected with probability of 0.1. 

For estimation I used Propensity Score Matching that ensures treated and control groups are comparable based on observed covariates. The propensity score represents the probability of receiving the treatment, given the covariates, and is estimated using logistic regression. It is estimated using logistic regression, then k-nearest neighbors is applied to match treated units with the most similar control units based on their propensity scores. A caliper threshold ensures that only close matches are retained, reducing poor-quality pairings. 

### Questions?
1. I expected higher bias in the case of SBM but its significantly lower. Maybe blocks just catch network interference effect?
2. Not sure what to do next - real data? other features? other methods (IPW)?

### Next steps?
[] Re-do SBM? Use other model?
[] In reality we won't have a true network, should I simulate uncertainty here, "corrupt network"?