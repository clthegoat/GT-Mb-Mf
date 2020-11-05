import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
import torch.nn as nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import numdifftools as nd
import time
import torch.autograd.functional as af



def lqr_controller(T, F, f, C, c, V_T, v_T):

    """
    lqr controller:
    for a typical control problem, need to specify:
    * input dim: x:n, u:m
    * dynamic function (linear) F_t, f_t: x_t+1 = F_t*[x_t,u_t]^T + f_t
    * F: T*n*(m+n), f: T*n*1
    * cost function (linear)  C_t, c_t: r_t = 0.5*[x_t,u_t]*C_t*[x_t,u_t]^T + c_t*[x_t,u_t]^T
    * C: T*(m+n)*(m+n), c: T*(m+n)*1
    * terminal cost (quadratic) V, v: r(x_T) = 0.5*x_T^T*V*x_T + v*x_T^T
    * V: n*n, v: n*1
    * horizon length T
    
    functions:
    Output: K_t, k_t
    K: T*m*n, k: T*m*1

    """
    
    n = v_T.shape[0]
    m = F.shape[2]-n

    K = np.zeros((T,m,n))
    k = np.zeros((T,m,1))
    
    
    V = V_T
    v = v_T

    #main loop
    for t in range(T-1,-1,-1):
        #initialize Q,q from T-1:
        Q = C[t,:,:] + np.dot(F[t,:,:].T,np.dot(V,F[t,:,:]))
        q = c[t,:,:] + np.dot(F[t,:,:].T,np.dot(V,f[t,:,:])) + np.dot(F[t,:,:].T,v)

        # divide Q into blocks
        Qxx = Q[0:n,0:n]
        Quu = Q[n:m+n,n:m+n]
        Qxu = Q[0:n,n:m+n]
        Qux = Q[n:m+n,0:n]
        qx = q[0:n,:]
        qu = q[n:m+n,:]

        #compute optimal K
        Kt = np.dot(-np.linalg.pinv(Quu+1e-4*np.eye(m)), Qux)
        kt = np.dot(-np.linalg.pinv(Quu+1e-4*np.eye(m)), qu)
        K[t,:,:] = Kt
        k[t,:,:] = kt

        #update V
        V = Qxx + np.dot(Qxu,Kt) + np.dot(Kt.T,Qux) + np.dot(np.dot(Kt.T,Quu),Kt)
        v = qx + np.dot(Qxu,kt) + np.dot(Kt.T,qu) + np.dot(np.dot(Kt.T,Quu),kt)

    return K, k


#may not used
def grad(net, inputs, eps=1e-4):
    '''
    inputs: N*a
    net: R^a -> R^b
    return: nabla_f: N*a*b
    '''
    #convert to torch
    inputs = torch.from_numpy(inputs).float()

    assert(inputs.ndimension() == 2)
    #get batch size, dim size
    nBatch, nDim = inputs.size()
    
    #x+,x-
    xp, xn = [], []
    e = 0.5*eps*torch.eye(nDim).type_as(inputs.data)
    for b in range(nBatch):
        for i in range(nDim):
            xp.append((inputs.data[b].clone()+e[i]).unsqueeze(0))
            xn.append((inputs.data[b].clone()-e[i]).unsqueeze(0))
    #xp,xn: (N*a)*a
    xs = Variable(torch.cat(xp+xn))
    
    #xs: (N*2a)*a
    fs = net(xs)
    #fs: (N*2a)*b
    
    
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fs_p, fs_n = torch.split(fs, nBatch*nDim) 
    g = ((fs_p-fs_n)/eps).view(nBatch, nDim, fDim).squeeze(2) #N*a*b

    #convert to numpy
    return g.detach().numpy()


# not accurate, maynot used
def num_hess(net, inputs, eps=1e-3):
    '''
    inputs: N*a
    net: R^a -> R^b
    return: nabla^2_f: N*a*a
    '''
    inputs = torch.from_numpy(inputs).float()

    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xpp, xpn, xnp, xnn = [], [], [], []
    e = eps*torch.eye(nDim).type_as(inputs.data)
    for b,i,j in itertools.product(range(nBatch), range(nDim), range(nDim)):
        xpp.append((inputs.data[b].clone()+e[i]+e[j]).unsqueeze(0))
        xpn.append((inputs.data[b].clone()+e[i]-e[j]).unsqueeze(0))
        xnp.append((inputs.data[b].clone()-e[i]+e[j]).unsqueeze(0))
        xnn.append((inputs.data[b].clone()-e[i]-e[j]).unsqueeze(0))
    #xs: (N*4a*a)*a
    xs = Variable(torch.cat(xpp+xpn+xnp+xnn))
    #fs: (N*4a*a)*b
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    #fpp: (N*a*a)*b
    fpp, fpn, fnp, fnn = torch.split(fs, nBatch*nDim*nDim)
    h = ((fpp-fpn-fnp+fnn)/(4*eps*eps)).view(nBatch, nDim, nDim, fDim).squeeze(3)
    # (N*a*a)
    return h.detach().numpy()



def compute_jacobian(f, x, output_dims):
    '''
    Normal:
        f: input_dims -> output_dims
    Jacobian mode:
        f: output_dims x input_dims -> output_dims x output_dims
    '''
    repeat_dims = tuple(output_dims) + (1,) * len(x.shape)
    jac_x = x.detach().repeat(*repeat_dims)
    jac_x.requires_grad_()
    jac_y = f(jac_x)
    ml = torch.meshgrid([torch.arange(dim) for dim in output_dims])
    index = [m.flatten() for m in ml]
    gradient = torch.zeros(output_dims + output_dims)
    gradient.__setitem__(tuple(index)*2, 1)
    
    jac_y.backward(gradient)
        
    return jac_x.grad.data


def torch_hessian(f, x):
    batch_size = x.shape[0]
    x_dim = x.shape[1]
    x.requires_grad_(True)
    loss = f(x)
    first_drv = torch.zeros(batch_size, x_dim)
    hessian = torch.zeros(batch_size, x_dim, x_dim)
    for n in range(batch_size):
        first_drv[n] = torch.autograd.grad(loss[n], x,
                                                    create_graph=True, retain_graph=True)[0][n]
        for i in range(x_dim):
            hessian[n][i] = torch.autograd.grad(first_drv[n][i], x,
                                                    create_graph=True, retain_graph=True)[0][n]
    
    return hessian.detach().numpy()



def forward_sim(X_0,U,f,c,v):
    '''
    simulate a traj given input
    X_0: init state n*1
    U: seq of inputs T*m*1
    f: dynamic
    c: cost_f
    v: val_f
    '''
    T = U.shape[0]
    n = X_0.shape[0]
    X = np.zeros((T+1,n,1))
    X[0,:,:] = X_0
    cost = 0
    for t in range(T):
        xu = torch.from_numpy(np.concatenate([X[t,:,:],U[t,:,:]])[:,0]).float()

        #forward dyn
        X[t+1,:,:] = f(xu).detach().numpy().reshape((n,1))
        cost += c(xu)
    cost += v(torch.from_numpy(X[T,:,0]).float())
    
    return X, cost






class ilqr_controller():

    """
    ilqr controller:
    for a typical control problem, need to specify:
    * dynamic function f(x_t,u_t)
    * reward function r(x_t,u_t)
    * terminal cost v(x_t)
    * horizon length T
    * an lqr controller
    * number of iteration M
    * initial trajectory x_0:T, u_0:T-1


    functions:
    * set initial trajecory
    * linearize
    * take gradient
    * call lqr
    * forward simulation


    """

    def __init__(self, X, U, up_X, low_X, up_U, low_U, T, dyn_f, cost_f, val_f, lr, M, visualize=0, print_loss=0):
        '''
        X: states, T+1*n*1
        U: actions T*m*1
        up_X: upperbound of x: n*1
        low_X: lower bound of x: n*1
        up_U: upperbound of u: m*1
        low_U: lower bound of u: m*1
        XU: concatenation of X_0:T-1, U_0:T-1, T*(m+n)*1
        dyn_f: transition dynamics R^(m+n) -> R^n, torch.nn.Module
        cost_f: reward function R^(m+n) -> R, torch.nn.Module
        val_f: value function R^n -> R, torch
        '''
        self.up_X = up_X
        self.low_X = low_X
        self.up_U = up_U
        self.low_U = low_U
        self.n = X.shape[1]
        self.m = U.shape[1]
        
        self.X = X #current state trajectory
        self.U = U #current input trajectory
        #clip the value to the boundary
        np.clip(self.X, self.low_X, self.up_X)
        np.clip(self.U, self.low_U, self.up_U)
        self.XU = np.concatenate([self.X[0:T,:,:],self.U],axis=1)
        self.T = T

        self.dyn_f = dyn_f
        self.cost_f = cost_f
        self.val_f = val_f

        self.iter_num = M
        self.lr = lr

        self.print_loss = print_loss
        self.visualize = visualize

        #self.map = Pool().map


    def hessian_torch(self, XU):
        return af.hessian(self.cost_f, XU).detach().numpy()

        
    
    # linearize dynamic and reward function
    def linearize(self):
        # linearize dynamics
        XU_t = torch.from_numpy(self.XU[:,:,0]).float()
        #print(XU_t.size())
        # apply num method
        #self.F = grad(self.dyn_f, self.XU[:,:,0]).reshape((self.T, self.n, self.n+self.m)) #T*n*n+m
        self.F = compute_jacobian(self.dyn_f, XU_t, (self.T,self.n))
        
        self.F = np.concatenate([self.F[i,:,i,:] for i in range(self.T)]).reshape((self.T, self.n, self.n+self.m))
        #print(self.F)
        self.f = self.dyn_f(XU_t).detach().numpy()
        self.f = self.f.reshape((self.T, -1, 1))
        #print(self.f)
        
        # linearize cost
        # TODO: can we further speedup with high accuracy?
        self.C = [self.hessian_torch(XU_t[i,:]) for i in range(self.T)]
        # self.C = Parallel(n_jobs=4,backend="threading")(delayed(self.hessian_torch)(XU_t[i,:]) for i in range(self.T))
        # self.C = self.map(self.hessian_torch, [XU_t[i,:] for i in range(self.T)])
        self.C = np.asarray(self.C)
        
        #self.C = torch_hessian(self.cost_f, XU_t)
        #try numerical
        #self.C = hess(self.cost_f, self.XU[:,:,0])
        #self.c = grad(self.cost_f, self.XU[:,:,0])
        
        self.c = compute_jacobian(self.cost_f, XU_t, (self.T,1))
        self.c = np.concatenate([self.c[i,:,i,:] for i in range(self.T)])
        #print(self.c.shape)
        #debug
        self.c = self.c.reshape((self.T, -1, 1))
        

        # linearize val
        X_T = torch.from_numpy(self.X[self.T,:,0]).float()
        #print(X_T.size())
        self.V_T = af.hessian(self.val_f, X_T).detach().numpy()
        self.v_T = compute_jacobian(self.val_f, X_T.view(1,-1), (1,))
        #self.v_T = grad(self.val_f,self.X[self.T,:,:].reshape((1,-1)))
        #debug
        self.v_T = self.v_T.reshape((-1, 1))
        # print(self.V_T.shape)
        # print(self.v_T.shape)

    

    # apply ilqr
    def solve_ilqr(self):
        '''
        solve the defined ilqr problem
        return: 
        local linear policy K, k
        last updated trajectory X, U
        last reward sequence
        last value function of last state
        '''

        for i in range(self.iter_num):
            
            R = np.zeros((self.T))
            #linearize
            self.linearize()
            

            #solve lqr
            self.K,self.k = lqr_controller(self.T, self.F, self.f, self.C, self.c, self.V_T, self.v_T)
            
            #forward propgate
            total_cost = 0
            new_X = self.X
            new_U = self.U
            for t in range(self.T):
                dx = new_X[t,:,:]-self.X[t,:,:]
                new_U[t,:,:] = self.lr*(np.dot(self.K[t,:,:],dx) + self.k[t,:,:]) + self.U[t,:,:]
                new_U[t,:,:] = np.clip(new_U[t,:,:], self.low_U, self.up_U)
                xu = torch.from_numpy(np.concatenate([new_X[t,:,:],new_U[t,:,:]])[:,0]).float()

                #forward dyn
                new_X[t+1,:,:] = self.dyn_f(xu).detach().numpy().reshape((self.n,1))
                new_X[t+1,:,:] = np.clip(new_X[t+1,:,:], self.low_X, self.up_X)
                
                #compute cost
                R[t]=self.cost_f(xu).detach().numpy()
                total_cost+=R[t]

            if self.print_loss:
                print(total_cost)
            #regain XU
            self.X = np.clip(new_X, self.low_X, self.up_X)
            self.U = np.clip(new_U, self.low_U, self.up_U)
            self.XU = np.concatenate([self.X[0:self.T,:,:],self.U[:,:,:]],axis=1) 

            #get value of last step (for computation of Q_target)
            last_X = torch.from_numpy(self.X[self.T,:,:]).float()
            
            last_value = self.val_f(last_X[:,0]).detach().numpy()

            #plot trajectory
            if self.visualize:
                if i%20==19:
                    plt.plot(self.X[:,0,0])
        if self.visualize:
            plt.show()
            
            

        

        return self.K, self.k, self.X, self.U, R, last_value
             

    


    

###test lqr:
##zero regulation
def test_lqr():
    T = 30
    F = np.ones((T,1,2)) #x_t+1 = x_t + u_t
    f = np.zeros((T,1,1))
    C = np.zeros((T,2,2)) #C_t = 0.5*x_t^2 + u_t^2+ 10x_t
    for i in range(T):
        C[i,:,:] = np.identity(2)

    c = np.zeros((T,2,1))
    for i in range(T):
        c[i,0,0] = 10
    V_T = np.ones((1,1))
    v_T = 10*np.ones((1,1))

    K,k = lqr_controller(T, F, f, C, c, V_T, v_T)
    #simulate
    X = np.zeros((T+1,1,1))
    X[0,:,:] = 50
    for i in range(T):
        u = np.dot(K[i,:,:],X[i,:,:])+k[i,:,:]
        X[i+1,:,:] = X[i,:,:] + u

    #visualize
    plt.plot(X[:,0,0])
    plt.show()

    ##double itegrator
    T = 10
    n = 2
    m = 1
    F = np.zeros((T,n,m+n))
    f = np.zeros((T,n,1))
    C = np.zeros((T,m+n,m+n))
    c = 0*np.ones((T,m+n,1))
    V_T = np.ones((n,n))
    v_T = np.zeros((n,1))
    for i in range(T):
        F[i,:,:] = np.asarray([[1,1,0],[0,1,1]])
        C[i,:,:] = np.identity(m+n)
        C[i,2,2] = 10

    K,k = lqr_controller(T, F, f, C, c, V_T, v_T)
    #simulate
    X = np.zeros((T+1,n,1))
    X[0,:,:] = np.asarray([[50],[0]])
    for i in range(T):
        u = np.dot(K[i,:,:],X[i,:,:])+k[i,:,:]
        X[i+1,:,:] = np.dot(F[i,:,:],np.concatenate([X[i,:,:],u]))

    #visualize
    plt.plot(X[:,0,0])
    plt.show()





##test gradient functions
def test_grad():

    #try with a model
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.linear2 = nn.Linear(2, 1)
            self.linear2.weight.data.fill_(0.0)
            self.linear2.weight[0,0] = 1.
            self.linear2.weight[0,1] = 1.
            self.linear2.weight[1,1] = 1.
            self.linear2.weight[1,2] = 1.
            


        def forward(self, x):
            pax_predict = self.linear2(x)
            #print(self.linear2.weight.data)
        
            return pax_predict

    f_t = Network()

    def fun(x):

        t_x = torch.from_numpy(x).float()
        
        f_x = f_t(t_x).detach().numpy()
        #print(f_x.shape)
        return f_x
        

    ##1d square
    #torch
    time_start = time.clock()
    model =Network()
    x = torch.ones((5,2))
    print([af.jacobian(model,x[i,:]) for i in range(x.shape[0])])
    print([af.hessian(model,x[i,:]) for i in range(x.shape[0])])
    time_e = time.clock()-time_start
    print(time_e)

    #numerical
    time_start = time.clock()
    model =Network()
    x = np.ones((5,2))
    df = nd.Gradient(fun)
    H = nd.Hessian(fun)
    print(list(map(df,x.tolist())))
    print(list(map(H,x.tolist())))
    time_e = time.clock()-time_start
    print(time_e)

    #from mpc 
    time_start = time.clock()
    model =Network()
    x = np.ones((5,2))
    print(grad(model,x))
    x = torch.ones((5,2))
    print([af.hessian(model,x[i,:]) for i in range(x.shape[0])])
    time_e = time.clock()-time_start
    print(time_e)

#define a dynamic function
class dynamic(nn.Module):
    #double integrater
    def __init__(self):
        super(dynamic, self).__init__()
        self.linear2 = nn.Linear(3, 2)
        self.linear2.weight.data.fill_(0.0)
        self.linear2.weight[0,0] = 1.
        self.linear2.weight[0,1] = 1.
        self.linear2.weight[1,1] = 1.
        self.linear2.weight[1,2] = 1.
        
    #print(xu.shape)
    def forward(self, xu):
        return self.linear2(xu)

#define a cost function
class cost(nn.Module):

    def forward(self, xu):
        return torch.square(torch.norm(xu+2,dim=len(xu.size())-1))
    

#define a terminal function
class val(nn.Module):

    def forward(self, x):
        return torch.square(torch.norm(x+2,dim=len(x.size())-1))

def test_ilqr():
    

    #give initial value
    T = 20
    n = 2
    X = np.ones((T+1,2,1))
    U = np.zeros((T,1,1))
    up_X = 10
    low_X = -10
    up_U = 1
    low_U = -1
    dyn_f = dynamic()
    cost_f = cost()
    val_f = val()
    #example usage:
    time_start = time.clock()
    ilqr_ctrl = ilqr_controller(X,U,up_X,low_X,up_U,low_U,T,dyn_f,cost_f,val_f, 0.5, 10, 1, 1)
    ilqr_ctrl.solve_ilqr()
    time_e = time.clock()-time_start
    print(time_e)





#test_lqr()
#test_grad()
#test_ilqr()
