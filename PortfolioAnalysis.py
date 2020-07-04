import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
plt.style.use("fivethirtyeight")

class PortFolAnalysis:
    def __init__(self, df):
        self.df = df
        self.returns = df.pct_change()
        self.mean_returns = self.returns.mean()
        self.cov_mat = self.returns.cov()
        self.return_arr = np.asarray(self.mean_returns)
        self.cov_matrix = np.asmatrix(self.cov_mat)
        self.ticker = df.columns
        
    def __annual_performance(self, weights):
        returns = np.sum(self.mean_returns*weights ) *252
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_mat, weights))) * np.sqrt(252)
        return std, returns
        
    def random_portfolios(self, num_portfolios=1000, risk_free_rate=0.0178):
        results = np.zeros((3, num_portfolios))
        alphs = tuple([1/len(self.ticker)]*len(self.ticker)) 
        weights_cache = np.random.dirichlet(alphs, num_portfolios)  
        for i in range(num_portfolios):
            weights = weights_cache[i]
            portfolio_std_dev, portfolio_return = PortFolAnalysis.__annual_performance(self, weights)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        #return results, weights_cache
        params, rand_portf  = results, weights_cache
        print(len(rand_portf), len(params))
        idx_min_vol = params[0].argmin()
        idx_max_retrn = params[1].argmax()
        idx_max_sharpe = params[2].argmax()
        min_vol_portf = rand_portf[idx_min_vol]
        max_return_portf = rand_portf[idx_max_retrn]
        max_sharpe_portf = rand_portf[idx_max_sharpe]
        print("Weigths of minimum volatility portfolio are: ", min_vol_portf, " and value of volatility is: ", params[0][idx_min_vol])
        print("Weigths of maximum return portfolio are: ", max_return_portf, "  and return is: ", params[1][idx_max_retrn])
        print("Weigths of maximum sharpe portfolio are: ", max_sharpe_portf, "  and sharpe is: ", params[2][idx_max_sharpe])
        
        plt.figure(figsize=(16, 10))
        plt.scatter(params[0], params[1], c = params[2], cmap = "YlGnBu", marker = "o", alpha=0.9)
        plt.ylabel("Return")
        plt.xlabel("Volatility")
        #plt.plot( params[1][idx_max_retrn], params[0][idx_max_retrn], 'g')
        plt.plot( params[0][idx_max_retrn],params[1][idx_max_retrn],  'g*')
        plt.annotate("  Maximum Return",(params[0][idx_max_retrn], params[1][idx_max_retrn]))
        plt.plot(params[0][idx_min_vol],params[1][idx_min_vol], 'r+')
        plt.annotate("  Minimum Volatility",( params[0][idx_min_vol], params[1][idx_min_vol]))
        plt.plot(params[0][idx_max_sharpe], params[1][idx_max_sharpe],'yo')
        plt.annotate("  Maximum Sharpe",( params[0][idx_max_sharpe], params[1][idx_max_sharpe]))
        plt.title("Visualization of Return vs. Volatility")
        plt.savefig('Return vs volatility.png')
        
    def Optimal_weights(self, req_return):
        n = len(self.ticker)
        symbols = self.ticker
        print(n)
        x = Variable(n)
        ret = (self.return_arr.T*x)*252 
        risk = quad_form(x, self.cov_matrix)*252
        prob = Problem(Minimize(risk), [sum(x)==1, ret >= req_return, x >= 0])
        prob.solve()
        print ("Optimal portfolio")
        print ("----------------------")
        for s in range(len(symbols)):
            print (" Investment in {} : {}% of the portfolio".format(symbols[s],round(100*x.value[s],2)))
        print ("----------------------")
        print ("Exp ret = {}%".format(round(100*ret.value,2)))
        print ("Expected risk    = {}%".format(round(100*risk.value**0.5,2)))
        return x.value, risk.value