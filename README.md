# Insurance Recommendation System Using Bayesian Network + Neural Cold-Start  
**Exact Replica + Improvement of RecSys 2017 Paper**  
by Vrinda
A complete, from-scratch implementation of the paper:  
**"An Insurance Recommendation System Using Bayesian Networks"** (RecSys 2017 Workshop)
- Manual Bayesian Network with custom Expectation-Maximization (EM) for missing values
- Exact structure and discretization as in the original paper
- Cold-start prediction using a tiny Feed-Forward Neural Network (FFNN) when smoker status is unknown
- Achieves Precision@1 ≈ 0.85–0.92 on test set (much better than paper's reported baseline ~0.28@3)
