import numpy as np
from scipy.stats import poisson,binom,norm
import matplotlib.pyplot as plt
from PHC import HC

def estimate_poisson(corpus):
    return np.mean(corpus,axis=0)


def phc(x,y):
    lam_1 = estimate_poisson(x)
    lam_2 = estimate_poisson(y)
    sx = np.std(x,axis=0)
    sy = np.std(y,axis=0)
    z = (lam_1 - lam_2)/np.sqrt((sx**2/x.shape[0]) + (sy**2/y.shape[0]))
    pvals = 1 - norm.cdf(z)
    hc, i_star = HC(pvals)
    return np.where(pvals < np.sort(pvals)[i_star], 1,0)

if __name__ == '__main__':
    n = 100
    m = 100
    p = 1000
    s = 0.01


    fig = plt.figure(figsize=(5, 1.5))
    gs = plt.GridSpec(3, 2)

    # Needed to add spacing between 1st and 2nd plots
    # Add a margin between the main title and sub-plots
    fig.subplots_adjust(wspace=0.2, top=0.85)
    #fig.suptitle("Main Title", fontsize=15)
    fig.text(0.5, 0.04, '% increase of l1 on the subset of discriminative words', ha='center', va='center',fontsize=12)
    #fig.text(0.06, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')

    # Add the subplots
    ax = []



    deltas = np.arange(5,22,2)
    for i,l2 in enumerate([5,10,15,20,25,30]):
        ax.append(fig.add_subplot(gs[i]))
        c = []
        d = []
        for delta in deltas:
            l1 = l2*(1+delta/100)
            y = np.random.poisson(lam=l2,size=(m,p))
            x = np.concatenate((np.random.poisson(lam=l1,size = (n,int(s*p))),np.random.poisson(lam=l2,size = (n,p-int(s*p)))),axis=1)
            hc = phc(x,y)

            c.append(sum(hc[:int(s*p)])/int(s*p))
            d.append(sum(hc[int(s*p):])/(p-int(s*p)))
        
        ax[i].plot(deltas,c,label="Fraction of important features selected")
        ax[i].plot(deltas,d,label="Fraction of non-important features selected")
    
        ax[i].title.set_text(f'l2 = {l2}')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize = 12)

    #plt.figure(figsize=(1,2.5))
    plt.show()





    """
    deltas = np.arange(5,22,2)
    for i,l2 in enumerate([5,10,15,20,25,30]):
        c = []
        d = []
        for delta in deltas:
            l1 = l2*(1+delta/100)
            x = np.random.poisson(lam=l1,size=(n,p))
            y = np.concatenate((np.random.poisson(lam=l2,size = (m,int(s*p))),np.random.poisson(lam=l1,size = (m,p-int(s*p)))),axis=1)
            hc = phc(x,y)

            c.append(sum(hc[:int(s*p)])/int(s*p))
            d.append(sum(hc[int(s*p):])/(p-int(s*p)))
        
        plt.subplot(3,2,i+1)
        plt.plot(deltas,c,label="proportion of important features detected ")
        plt.plot(deltas,d,label="proportion of non-important features wrongly detected ")
        plt.title(f"Lambda = {l2}")
        plt.legend()
    plt.show()"""