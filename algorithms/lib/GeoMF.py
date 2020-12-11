import geo_utils
from scipy import sparse
class GeoMF:
    def __init__(self,K,delta,gamma,epsilon,lambda_,max_iters,grid_distance):
        self.K = K
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.max_iters = max_iters
        self.grid_distance = grid_distance
        self.data = object()

    def train(self,training_matrix, poi_coos):
        C = sparse.csr_matrix(training_matrix)
        M = training_matrix.shape[0] # users
        N = training_matrix.shape[1] # items
        W = C.copy()
        W = np.log10(1+W*10**self.epsilon)

        C = C>0

        P = np.random.rand(M,self.K)
        Q = np.random.rand(N,self.K)
        X = np.random.rand(M,self.lat_grid_num*self.lon_grid_num)
        Y = np.random.rand(N,self.lat_grid_num*self.lon_grid_num)
        lat_lon = np.array([v for k, v in poi_coos.items()])
        min_lat = np.min(lat_lon[:,0])
        max_lat = np.max(lat_lon[:,0])
        min_lon = np.min(lat_lon[:,1])
        max_lon = np.max(lat_lon[:,1])
        

        W_u = dict()
        for i in range(M):
            W_u[i] = sparse.spdiags(W[i,:].A,diags=0,m=N,n=N)

        W_i = dict()
        for i in range(N):
            W_i[i] = sparse.spdiags(W[:,i].A,diags=0,m=M,n=M)
            
        dist_lat = geo_utils.haversine(min_lat,min_lon,max_lat,min_lon)
        dist_lon = geo_utils.haversine(min_lat,min_lon,min_lat,max_lon)

        num_grids_lat = np.ceil((max_lat-min_lat)/self.grid_distance)
        num_grids_lon = np.ceil((max_lon-min_lon)/self.grid_distance)
        for i in range(N):
            poi_coo = poi_coos[i,:]
            for lat_grid in range(num_grids_lat): 
                for lon_grid in range(num_grids_lon):
                    grid_lat = (lat_grid+0.5)*self.grid_distance + lat_min
                    grid_lon = (lon_grid+0.5)*self.grid_distance + lon_min
                    dist = geo_utils.haversine(poi_coo[0],poi_coo[1],grid_lat,grid_lon)
                    if dist < self.delta:
                        Y[i,lat_grid*num_grids_lon + lon_grid] = np.exp(-dist/1.5)

        self.optimize_latent_factors(M,n,W_u,W_i,P,Q,C,X,Y)
        Ct = C.T
        Wt = W.T
        Yt = Y.T
        YtY = Yt @ Y
        Xt = self.optimize_activity(M,Ct, Wt, Xt, Yt, YtY, Ut, V)
        X = Xt.T
        X[X < 0] = 0
        self.data.X = X
        self.data.Y = Y
        self.data.P = P
        self.data.Q = Q

    def optimize_latent_factors(self,M,n,W_u,W_i,P,Q,C,X,Y):
        Im = sparse.identity(M)
        In = sparse.identity(n)
        Ik = sparse.identity(self.K)
        for i in range(self.max_iters):
            QtQ = Q.T @ Q
            for j in range(M):
                P[j] = sparse.linalg.inverse(Q.T@(W_u[j]-In)+QtQ+self.gamma*Ik) @ Q.T @ W_u[j] @ (C[j,:] -Y @ X[j])


            PtP = P.T @ P
            for j in range(N):
                Q[j] = sparse.linalg.inverse(P.T@(W_i[j]-Im)+PtP+self.gamma*Ik) @ P.T @ W_i[j] @ (C[:,j] -X @ Y[j])


    def optimize_activity(self,M, Ct, Wt, Xt, Yt, YtY, Ut, V):
        reg = self.lambda_
        YtV = Yt @ V
        L, M = Xt.shape
        user_cell = []
        item_cell = []
        val_cell = []
        for i in range(M):
            w = Wt[:,i]
            r = Ct[:,i]
            Ind = w>0
            wi = w[Ind]
            ri = r[Ind]

            if(Ind.getnnz() == 0):
                Wi = np.zeros(0)
            else:
                Wi = sparse.spdiags(sqrt(wi),0,len(wi),len(wi))

            subYt = Yt[:, Ind]
            subV = V[Ind, :]
            YC = subYt @ Wi
            grad_invariant =  YC @ (sqrt(wi) * (subV @ Ut[:,i])) - subYt @ (wi * ri + ri) + YtV @ Ut[:,i] + reg;
            J = range(1,len(grad_invariant)+1)
            ind = grad_invariant <= 0
            grad_invariant = sparse.csr_matrix(J[ind], 1, grad_invariant[ind], len(grad_invariant), 1);
        
            x = self.line_search(YC, YtY, grad_invariant, Xt[:,i])

            # [loc, I, val ] = find(x);
            user_cell[i] = i * I;
            item_cell[i] = loc;
            val_cell[i] = val;
       # P = rand(M, K);
       # Q = rand(N, K);
       
       # X = rand(M, lat_grid_num * lng_grid_num);
       # Y = zeros(N, lat_grid_num * lng_grid_num);

       
    def line_search(self,YC, YtY, grad_i, x)
        alpha = 1
        beta = 0.1
        for iter in range(1,6):
            grad = grad_i + YC @ (x.T @ YC).T + YtY * x;
            J = range(1,len(grad)+1)
            Ind = grad < 0 || x > 0
            grad = sparse.csr_matrix(J(Ind), 1, grad(Ind), len(grad), 1)
            for step in range(1,11): # search step size
                xn = max(x - alpha * grad, 0)
                d = xn - x
                dt = d.T
                gradd = dt @ grad
                dyc = dt @ YC
                dQd = dt @ (YtY @ d) + dyc @ dyc.T
                suff_decr = 0.99 @ gradd + 0.5 @ dQd < 0
                if step == 1:
                    decr_alpha = ~suff_decr; xp = x;

                if decr_alpha
                    if suff_decr:
                        x = xn
                        break
                    else:
                        alpha = alpha * beta
                else:
                    if ~suff_decr || (xp != xn).getnnz()==0:
                        x = xp
                        break
                    else:
                        alpha = alpha / beta
                        xp = xn
        return x

    def predict(self, uid, lid):
        return self.data.P[uid, :].dot(self.data.Q[lid, :]) + self.data.X[uid, :].dot(self.data.Y[lid, :])
