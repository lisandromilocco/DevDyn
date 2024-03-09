import numpy as np
import matplotlib.pyplot as pl
import matplotlib
matplotlib.use('Agg')


def calculate_jacobian(A_1D, B_1D, n, Da, Db, F, K):
    # Initialize Jacobian matrices
    J11 = np.zeros((n * n, n * n),dtype=float)
    J22 = np.zeros((n * n, n * n),dtype=float)

    V12 = np.zeros(n * n,dtype=float)
    V21 = np.zeros(n * n,dtype=float)

    # Calculate Jacobian matrices
    for i in range(n):
        for j in range(n):
            row = i * n + j  # Convert (i, j) coordinates to row index

            if i > 0:
                J11[row, row - n] = Da
                J22[row, row - n] = Db
            else:
                J11[row, (n - 1) * n + j] = Da
                J22[row, (n - 1) * n + j] = Db

            if i < n - 1:
                J11[row, row + n] = Da
                J22[row, row + n] = Db
            else:
                J11[row, j] = Da
                J22[row, j] = Db

            if j > 0:
                J11[row, row - 1] = Da
                J22[row, row - 1] = Db
            else:
                J11[row, i * n + n - 1] = Da
                J22[row, i * n + n - 1] = Db

            if j < n - 1:
                J11[row, row + 1] = Da
                J22[row, row + 1] = Db
            else:
                J11[row, i * n] = Da
                J22[row, i * n] = Db


    for i in range(n * n):
        J11[i, i] = -4 * Da - B_1D[0][i] ** 2 - F
        J22[i, i] = -4 * Db + 2 * A_1D[0][i] * B_1D[0][i] - K - F
        V12[i] = -2 * A_1D[0][i] * B_1D[0][i]
        V21[i] = B_1D[0][i] * B_1D[0][i]


    J12 = np.diag(V12)
    J21 = np.diag(V21)
    J = np.block([[J11, J12], [J21, J22]])

    return J



def s_k_update (s,J,A_1D,B_1D, n, DA, DB, f, k,delta_t):
    b_k=np.zeros(n * n * 2)
    for i in range(n * n):
        b_k[n * n + i] = -B_1D[0][i]  # Assuming B is a NumPy array of length n^2
    diff_s= ( np.matmul(J,s) + b_k ) * delta_t
    s += diff_s
    return s

def s_f_update (s,J,A_1D,B_1D, n, DA, DB, f, k,delta_t):
    b_f=np.zeros(n * n * 2)
    for i in range(n * n):
        b_f[i] = 1-A_1D[0][i]  # Assuming B is a NumPy array of length n^2
        b_f[n * n + i] = -B_1D[0][i]  # Assuming B is a NumPy array of length n^2
    diff_s= ( np.matmul(J,s) + b_f ) * delta_t
    s += diff_s
    return s

def discrete_laplacian(M):
    # Calculate Laplacian. Modified from github.com/benmaier/reaction-diffusion.

    L = -4 * M
    L += np.roll(M, (0, -1), (0, 1))  # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L


def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    # Update states according to Gray-Scott model. Modified from github.com/benmaier/reaction-diffusion.

    # Obtain Laplacian
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)

    # Update
    diff_A = (DA * LA - A * B ** 2 + f * (1 - A)) * delta_t
    diff_B = (DB * LB + A * B ** 2 - (k + f) * B) * delta_t

    A += diff_A
    B += diff_B

    return A, B

def develop2(initial, n, DA, DB, f, k, delta_t, max_steps):
    A = initial[0,:,:]
    B = initial[1,:,:]
    s_k = np.zeros(2*n**2)
    s_f = np.zeros(2*n**2)

    for t in range(max_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)

        A_1D = np.array(A.reshape(1, -1))
        B_1D = np.array(B.reshape(1, -1))
        J = calculate_jacobian(A_1D, B_1D, n, DA, DB, f, k)

        s_f = s_f_update(s_f,J,A_1D,B_1D, n, DA, DB, f, k, delta_t)
        s_k = s_k_update(s_k,J,A_1D,B_1D, n, DA, DB, f, k, delta_t)


        if (t % 100 == 0):
            np.savetxt('SIMULATE/s_k_' + str(t) + '.txt', s_k.reshape(1, -1), delimiter=',', fmt='%.16e')
            np.savetxt('SIMULATE/s_f_' + str(t) + '.txt', s_f.reshape(1, -1), delimiter=',', fmt='%.16e')

            s_k_2D = s_k[0:n ** 2].reshape(n, n)
            draw(s_k_2D, "SIMULATE/s_k_" + str(t) + ".png")

            s_f_2D = s_f[0:n ** 2].reshape(n, n)
            draw(s_f_2D, "SIMULATE/s_f_" + str(t) + ".png")

            draw(A, "SIMULATE/A_REF_" + str(t) + ".png")
            np.savetxt('SIMULATE/A_REF' + str(t) + '.txt', A.reshape(1, -1), delimiter=',', fmt='%.16e')

            #draw(B, "SIMULATE/B_REF_" + str(t) + ".png")
            #np.savetxt('SIMULATE/B_REF' + str(t) + '.txt', B.reshape(1, -1), delimiter=',', fmt='%.16e')


        print(t)

    output = [A, B, s_k, s_f]
    return output


def develop3(initial, n, DA, DB, f, k, delta_t, max_steps,name):
    A = initial[0,:,:]
    B = initial[1,:,:]
    s_k = np.zeros(2*n**2)

    for t in range(max_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)

        if (t % 100 == 0):
            draw(A, "SIMULATE/A_"+name+"_" + str(t) + ".png")
            np.savetxt('SIMULATE/A_'+name+"_" + str(t) + '.txt', A.reshape(1, -1), delimiter=',', fmt='%.16e')

            #draw(B, "SIMULATE/B_"+name+"_" + str(t) + ".png")
            #np.savetxt('SIMULATE/B_'+name+"_" + str(t) + '.txt', B.reshape(1, -1), delimiter=',', fmt='%.16e')


    output = [A, B, s_k]
    return output

def draw(A,filename):
    fig, ax = pl.subplots()
    ax.imshow(A, cmap="Greys", vmin=0, vmax=1)
    ax.set_title("A")
    ax.axis("off")
    pl.savefig(filename)
    pl.close(fig)
    #pl.show()

def read_initial(fileA,fileB):
    file_path = fileA
    with open(file_path, 'r') as file:
        data = file.readline().strip().split(',')  # Read a single line and split by ','
    data = [float(element) for element in data]
    A = np.array(data).reshape(50, 50)

    file_path = fileB
    with open(file_path, 'r') as file:
        data = file.readline().strip().split(',')  # Read a single line and split by ','
    data = [float(element) for element in data]
    B = np.array(data).reshape(50, 50)

    return A,B

if __name__ == '__main__':

    f_ref=0.032
    k_ref=0.060
    DA, DB, f, k = 0.32, 0.06, f_ref, k_ref  # bacteria
    n=50
    delta_t = 1e-1
    max_steps = int(5.1e3)

    # Run the nominal developmental trajectory and calculate sensitivities
    A, B = read_initial('SIMULATE/A0.txt','SIMULATE/B0.txt')
    initial_frame = np.array([A, B])
    final_frame = develop2(initial_frame, n, DA, DB, f, k, delta_t, max_steps)


    # Run a developmental trajectory with a perturbed value for f
    k = k_ref  # bacteria
    f = f_ref+2.0e-2

    A, B = read_initial('SIMULATE/A0.txt', 'SIMULATE/B0.txt')
    initial_frame = np.array([A, B])
    develop3(initial_frame, n, DA, DB, f, k, delta_t, max_steps,"f")

    # Run a developmental trajectory with a perturbed value for k
    k = k_ref-.8e-2
    f = f_ref

    A, B = read_initial('SIMULATE/A0.txt', 'SIMULATE/B0.txt')
    initial_frame = np.array([A, B])
    develop3(initial_frame, n, DA, DB, f, k, delta_t, max_steps, "k")