import geo_utils
from scipy import sparse
import numpy as np
from numba import njit, prange

@njit
def _set_y(Y,N,poi_coos,num_grids_lat,num_grids_lon,grid_distance,min_lat,min_lon,max_lat,max_lon,delta):
    for i in range(N):
        poi_coo = poi_coos[i]
        for lat_grid in range(num_grids_lat): 
            for lon_grid in range(num_grids_lon):
                grid_lat = (lat_grid+0.5)*grid_distance + min_lat
                grid_lon = (lon_grid+0.5)*grid_distance + min_lon
                dist = geo_utils.haversine(poi_coo[0],poi_coo[1],grid_lat,grid_lon)
                if dist < delta:
                    Y[i,lat_grid*num_grids_lon + lon_grid] = np.exp(-dist/1.5)
    return Y

class GeoMF:
    def __init__(self,K,delta,gamma,epsilon,lambda_,max_iters,grid_distance):
        self.K = K
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.max_iters = max_iters
        self.grid_distance = grid_distance
        self.data = {}

    def train(self,training_matrix, poi_coos):
        C = sparse.csr_matrix(training_matrix)
        # print(C)
        M = training_matrix.shape[0] # users
        N = training_matrix.shape[1] # items
        W = C.copy()
        # print(W)
        W[W!=0] = np.log(1+W[W!=0]*10**self.epsilon)
        # print(W)

        C = C>0

        P = np.random.rand(M,self.K)
        Q = np.random.rand(N,self.K)
        lat_lon = np.array([v for k, v in poi_coos.items()])
        min_lat = np.min(lat_lon[:,0])
        max_lat = np.max(lat_lon[:,0])
        min_lon = np.min(lat_lon[:,1])
        max_lon = np.max(lat_lon[:,1])
        
        dist_lat = geo_utils.haversine(min_lat,min_lon,max_lat,min_lon)
        dist_lon = geo_utils.haversine(min_lat,min_lon,min_lat,max_lon)

        num_grids_lat = int(np.ceil(dist_lat/self.grid_distance))
        num_grids_lon = int(np.ceil(dist_lon/self.grid_distance))
        print(f"num grids in lat: {num_grids_lat}, num grids in lon: {num_grids_lon}")

        X = np.random.rand(M,num_grids_lat*num_grids_lon)
        Y = np.zeros((N,num_grids_lat*num_grids_lon))

        W_u = dict()
        for i in range(M):
            W_u[i] = sparse.spdiags(W[i,:].A[0],diags=0,m=N,n=N)

        W_i = dict()
        for i in range(N):
            W_i[i] = sparse.spdiags(W[:,i].A[0],diags=0,m=M,n=M)

        print("Calculating areas transition probabilities")

        _set_y(Y,N,np.array([coo for k,coo in poi_coos.items()]),num_grids_lat,num_grids_lon,self.grid_distance,min_lat,min_lon,max_lat,max_lon,self.delta)
        # print(Y.max())
        # print(Y.min())
        # print(Y)
        # for i in range(N):
        #     poi_coo = poi_coos[i]
        #     for lat_grid in range(num_grids_lat): 
        #         for lon_grid in range(num_grids_lon):
        #             grid_lat = (lat_grid+0.5)*self.grid_distance + min_lat
        #             grid_lon = (lon_grid+0.5)*self.grid_distance + min_lon
        #             dist = geo_utils.haversine(poi_coo[0],poi_coo[1],grid_lat,grid_lon)
        #             if dist < self.delta:
        #                 Y[i,lat_grid*num_grids_lon + lon_grid] = np.exp(-dist/1.5)

        print("Optimizing latent factors")
        self.optimize_latent_factors(M,N,W_u,W_i,P,Q,C,X,Y)
        # print("P:")
        # print(P)
        # print("Q:")
        # print(Q)
        Ct = C.T
        Wt = W.T
        Yt = Y.T
        YtY = Yt @ Y
        print("Optimizing activity")
        Xt = self.optimize_activity(Ct, Wt, X.T, Yt, YtY, P.T, Q)
        X = Xt.T
        X[X < 0] = 0
        self.data['X'] = X
        self.data['Y'] = Y
        self.data['P'] = P
        self.data['Q'] = Q

    def optimize_latent_factors(self,M,N,W_u,W_i,P,Q,C,X,Y):
        Im = sparse.identity(M)
        In = sparse.identity(N)
        Ik = sparse.identity(self.K)
        for i in range(self.max_iters):
            print(f"iteration {i}")
            QtQ = Q.T @ Q
            for j in range(M):
                # print((Q.T @ W_u[j]).shape)
                # print((C[j,:] -Y @ X[j]).shape)
                # print((Q.T @ W_u[j] @ (C[j,:] -Y @ X[j]).T).shape)
                # print( np.linalg.inv(Q.T @ W_u[j] @ (C[j,:] -Y @ X[j]).T).shape)
                P[j] =np.linalg.inv(Q.T@(W_u[j]-In)@Q+QtQ+self.gamma*Ik) @ Q.T @ W_u[j] @ (C[j,:].A.flatten() -Y @ X[j,:])
                # P[j] = sparse.linalg.inv(Q.T@(W_u[j]-In)+QtQ+self.gamma*Ik) @ Q.T @ W_u[j] @ (C[j,:] -Y @ X[j])
                # print((Q.T @ W_u[j] @ Q + QtQ + self.gamma*Ik))
                # print((Q.T @ W_u[j] + Q.T))
                # print(C[j,:].T.A - (Y @ X[j,:].T))
                # print(((Q.T @ W_u[j] + Q.T) @ (C[j,:].T - Y @ X[j,:].T)))
                # print((Q.T @ W_u[j] @ Q + QtQ + self.gamma*Ik).shape)
                # print(((Q.T @ W_u[j] + Q.T) @ (C[j,:].T.A - Y @ X[j,:].T)).shape)
                # print(C[j,:].T.A.shape)
                # print((Y @ X[j,:].T).shape)
                # print((C[j,:].T.A - Y @ X[j,:].T).shape)
                # P[j] = np.linalg.solve((Q.T @ W_u[j] @ Q + QtQ + self.gamma*Ik),((Q.T @ W_u[j] + Q.T) @ (C[j,:].T.A - Y @ X[j,:].T)))


            PtP = P.T @ P
            for j in range(N):
                Q[j] =np.linalg.inv(P.T@(W_i[j]-Im)@P+PtP+self.gamma*Ik) @ P.T @ W_i[j] @ (C[:,j].A.flatten() -X @ Y[j,:])
                # Q[j] = sparse.linalg.inv(P.T@(W_i[j]-Im)+PtP+self.gamma*Ik) @ P.T @ W_i[j] @ (C[:,j] -X @ Y[j])
                # Q[j] = sparse.linalg.inv(P.T@(W_i[j]-Im)+PtP+self.gamma*Ik) @ P.T @ W_i[j] @ (C[:,j] -X @ Y[j])
                # Q[j] = np.linalg.solve((P.T @ W_i[j] @ P + PtP + self.gamma*Ik),((P.T @ W_i[j] + P.T) @ (C[:,j].T.A - X @ Y[j,:].T)))


    def optimize_activity(self, Ct, Wt, Xt, Yt, YtY, Ut, V):
        reg = self.lambda_
        YtV = Yt @ V
        L, M = Xt.shape
        user_cell = [None]*M
        item_cell = [None]*M
        val_cell = [None]*M
        for i in range(M):
            w = Wt[:,i].A.flatten()
            r = Ct[:,i].A.flatten()
            Ind = w>0
            wi = w[Ind]
            ri = r[Ind]

            if(np.count_nonzero(Ind) == 0):
                Wi = np.zeros(0)
                print("Ops...")
            else:
                Wi = sparse.spdiags(np.sqrt(wi),0,len(wi),len(wi))

            subYt = Yt[:, Ind]
            subV = V[Ind, :]
            YC = subYt @ Wi
            grad_invariant =  YC @ (np.sqrt(wi) * (subV @ Ut[:,i])) - subYt @ (wi * ri + ri) + YtV @ Ut[:,i] + reg;
            J = np.array(range(len(grad_invariant)))
            ind = grad_invariant <= 0
            # print(ind)
            # print(grad_invariant)
            # print(grad_invariant[ind])
            # print(J[ind])
            # print(np.repeat(1,len(grad_invariant)))
            grad_invariant = sparse.csr_matrix((grad_invariant[ind],(J[ind], np.repeat(0,len(grad_invariant[ind])))), (len(grad_invariant), 1));
            # print(grad_invariant)
        
            x = self.line_search(YC, YtY, grad_invariant, Xt[:,i])

            cols = np.nonzero(x)[0]
            
            # [loc, I, val ] = find(x);
            user_cell[i] = i * np.ones(len(cols))
            item_cell[i] = cols
            val_cell[i] = x[x!=0]

        Xt = sparse.csr_matrix((np.concatenate(val_cell),(np.concatenate(item_cell), np.concatenate(user_cell))), (L, M))
        return Xt
       
       # X = rand(M, lat_grid_num * lng_grid_num);
       # Y = zeros(N, lat_grid_num * lng_grid_num);

       
    def line_search(self,YC, YtY, grad_i, x):
        alpha = 1
        beta = 0.1
        for iter in range(1,6):
            # print(grad_i.shape)
            # print(YC.shape)
            # print((x.T @ YC).T.shape)
            # print(grad_i.shape)
            # print((x.T @ YC).T.shape)
            # print((YC @ (x.T @ YC).T + YtY @ x).shape)
            grad = grad_i.A.flatten() + YC @ (x.T @ YC).T + YtY @ x;
            J = np.array(range(len(grad)))
            Ind = (grad < 0) | (x > 0)
            # print(Ind.shape)
            # print(J[Ind].shape)
            # print(grad[Ind].shape)
            # print(np.repeat(1,len(J[Ind])).shape)

            grad = sparse.csr_matrix((grad[Ind], (J[Ind],np.repeat(0,len(J[Ind])))), (len(grad), 1)).A.flatten()
            # grad_invariant = sparse.csr_matrix((grad_invariant[ind],(J[ind], np.repeat(1,len(grad_invariant[ind])))), (len(grad_invariant), 1));
            for step in range(1,11): # search step size
                xn = np.maximum(x - alpha * grad, 0)
                d = xn - x
                dt = d.T
                gradd = dt @ grad
                dyc = dt @ YC
                dQd = dt @ (YtY @ d) + dyc @ dyc.T
                suff_decr = (0.99 * gradd + 0.5 * dQd) < 0
                if step == 1:
                    decr_alpha = ~suff_decr
                    xp = x

                if decr_alpha:
                    if suff_decr:
                        x = xn
                        break
                    else:
                        alpha = alpha * beta
                else:
                    if ~suff_decr | np.count_nonzero(xp != xn) == 0:
                        x = xp
                        break
                    else:
                        alpha = alpha / beta
                        xp = xn
        return x

    def predict(self, uid, lid):
        return self.data['P'][uid, :].dot(self.data['Q'][lid, :].T) + self.data['X'][uid, :].dot(self.data['Y'][lid, :].T)
