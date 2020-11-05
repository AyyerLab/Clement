import numpy as np


# affine input data
ins = np.array([[-100,-100,0], [0,100,0], [100,-100,0], [0.1,0.1,0.1]]) # <- primary system
#out = np.array([[-1000,-1000,0], [0,1000,0], [1000,-1000,0], [1,1,1]]) # <- secondary system (scale 10x)
out = np.array([[-1000,-1000,0], [0,1000,0], [1000,-999,0], [1,1,1]]) # <- secondary system (scale 10x and sheer)

# finding transformation
l = len(ins)
e = lambda r,d: np.linalg.det(np.delete(np.vstack([r, ins.T, np.ones(l)]), d, axis=0))
M = np.array([[(-1)**i * e(R, i) for R in out.T] for i in range(l+1)])
A, t = np.hsplit(M[1:].T/(-M[0])[:,None], [l-1])
t = np.transpose(t)[0]
# output transformation
print("Affine transformation matrix:\n", A)
print("Affine transformation translation vector:\n", t)


p = np.array([[1.0, 1.0, 1.0],[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
p_prime = np.array([[2.414213562373094,  5.732050807568877, 0.7320508075688767], [2.7677669529663684, 6.665063509461097, 0.6650635094610956],
            [2.7677669529663675, 5.665063509461096, 1.6650635094610962]])

Q = p[1:]- p[0]
Q_prime = p_prime[1:] - p_prime[0]




