import numpy as np

from simulator import generate_companies

from scipy.special import psi
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from gensim.matutils import mean_absolute_difference


class SimpleLDA:
    def __init__(self, word2id, 
                 industries, 
                 a=1.0, 
                 b=1.0, 
                 e_step_iter=50,
                 print_every=25,
                 estep_threshold=1e-5
                ):
        self.a = a
        self.b = b
        self.estep_threshold = estep_threshold
        
        self.word2id = word2id
        self.industries = industries
        id2word = {}
        for k,v in word2id.items():
            id2word[v] = k
        self.id2word = id2word
        self.e_step_iter = e_step_iter
        self.print_every = print_every
    
    def _e_step(self, X, metadata):
        import numpy
        from scipy.special import psi
        from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack
        
        a,b = self.a, self.b
        D,V = X.shape
        q_theta = np.random.gamma(1.0, size=(D,2))
        q_z = lil_matrix((D,V))
        phi = self.phi

        for industry in range(len(self.industries)):
            ind = metadata==industry
            q_theta_ind = q_theta[ind,:]
            X_ind = X[ind]
            q_z_ind = lil_matrix(X_ind.shape)
            d,v = X_ind.nonzero()
            q_theta_ind_old = q_theta_ind.copy()
            
            for i in range(self.e_step_iter):
                # Compute q(z)
                coef = np.clip(psi(q_theta_ind[d,-1])-psi(np.sum(q_theta_ind[d], axis=1)), a_min=-100.0, a_max=100.0)
                
                bg_w = np.exp(coef)*phi[-1,v]
                ind_w = np.exp(-coef)*phi[industry,v]
                q_z_ind[d,v] = bg_w/(bg_w+ind_w+1e-9)

                # Compute q(theta)
                q_theta_ind[:,0] = a+np.sum(q_z_ind.multiply(X_ind), axis=1).ravel()
                q_theta_ind[:,1] = b+np.sum(X_ind-q_z_ind.multiply(X_ind), axis=1).ravel()
                
                if mean_absolute_difference(q_theta_ind.ravel(), q_theta_ind_old.ravel()) <= self.estep_threshold:
                    break
                q_theta_ind_old = q_theta_ind.copy()

            q_z[ind] = q_z_ind
            q_theta[ind] = q_theta_ind
        
        return q_z, q_theta
    
    def _m_step(self, X, q_z, metadata):
        # Why PySpark? Whyyyyyyyy?
        import numpy
        from scipy.special import psi
        from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack
        
        industries = self.industries
        id2word = self.id2word
        _sstats = np.zeros(shape=(len(industries)+1, len(id2word)))

        for industry in range(len(industries)):
            ind = metadata==industry
            q_z_ind = q_z[ind]
            X_ind = X[ind]
            mlt = q_z_ind.multiply(X_ind)
            q_z_sm_bg = np.sum(mlt, axis=0)
            q_z_sm_ind = np.sum(X_ind - mlt, axis=0)

            # Background sstats
            _sstats[-1,:] = _sstats[-1,:] + q_z_sm_bg

            # Industry sstats
            _sstats[industry,:] = _sstats[industry,:] + q_z_sm_ind
        
        return _sstats
        
    def _update_phi(self):
        self.phi = self._sstats/(np.sum(self._sstats, axis=1, keepdims=True) + 1e-9)
        self._sstats = np.zeros(shape=(len(self.industries)+1, len(self.id2word)))

    def train(self, X, metadata, n_iter=50):
        industries = self.industries
        id2word = self.id2word
        self._sstats = np.zeros(shape=(len(industries)+1, len(id2word)))
        self.phi = np.random.dirichlet([1.0]*(len(id2word)), size=(len(industries)+1))

        for i in range(n_iter):
            if i%self.print_every == 0:
                print(i)
            q_z,_ = self._e_step(X, metadata)
            self._sstats = self._m_step(X, q_z, metadata)
            self._update_phi()
            
    def train_distributed(self, X_metadata_rdd, n_iter=50):
        industries = self.industries
        id2word = self.id2word
        self._sstats = np.zeros(shape=(len(industries)+1, len(id2word)))
        self.phi = np.random.dirichlet([1.0]*(len(id2word)), size=(len(industries)+1))
        
        for i in range(n_iter):
            print(i)
            self._sstats = (X_metadata_rdd
                             .mapPartitions(lambda u: [list(u)])
                             .map(lambda u: (vstack([el[0] for el in u], format="lil"), 
                                             np.array([el[1] for el in u])
                                             )
                                  )
                             .map(lambda line: (line[0], line[1], self._e_step(line[0], line[1])))
                             .map(lambda line: self._m_step(line[0], line[2][0], line[1]))
                                 .treeReduce(lambda a,b: a+b, depth=8)
                            )
            self._update_phi()