# Este script ha sido el codigo utilizado para resolver
# la primera practica de la asignatura de Fisica Solar y Clima Espacial

# Autora: Elena Arjona Galvez



import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


gtokg = 1e-3
cmtom = 1e-2
kmtom = 1e3
cmtokm = 1e-5


fil = 'model_jcd.dat'
model = np.loadtxt(fil,comments = '#')
z_m     = model[:,0] # [km]
P_m     = model[:,1]/cmtokm # [dym/cm2 = g/(cm s2)]  --> [g/ (km s2)]
rho_m   = model[:,2]/(cmtokm**3.) # [g/cm3] --> [g / km3]
T_m     = model[:,3] # [K]


gamma = 5./3.
g = 274.*1e-3 # [km/s2]


cs2_m = gamma*P_m/rho_m #[km2 / s2]
wc2_m = ((gamma*g)**2.)/(4.*cs2_m) #(km2/s4)/(km2/s2) = [s-2]
N2_m = g*(gamma - 1)/(gamma * (cs2_m/(gamma*g)))


def centered(f,h):
    dh = (h[-1]-h[0])/h.shape[0]
    return 0.5*(f[2:]-f[:-2])/dh

dcs_m = centered(cs2_m**0.5,z_m)
dwc_m = centered(wc2_m**0.5,z_m)
dN_m = centered(N2_m**0.5,z_m)

# =========================================
# ======= INTERPOLACION ===================
# =========================================


zint = interpolate.interp1d(wc2_m,z_m) # Esto nos servira para calcular la profundidad de corte.

cs2int = interpolate.interp1d(z_m,cs2_m)
wc2int = interpolate.interp1d(z_m,wc2_m)
N2int = interpolate.interp1d(z_m,N2_m)

dcsint = interpolate.interp1d(z_m[1:-1],dcs_m)
dwcint = interpolate.interp1d(z_m[1:-1],dwc_m)
dNint = interpolate.interp1d(z_m[1:-1],dN_m)


# ==========================================
# ====== DERIVADAS PARCIALES ===============
# ==========================================


def dFz(k_z,z,mode):
    cs2 = cs2int(z) ; N2 = N2int(z)
    wc2 = wc2int(z)

    dcs = dcsint(z) ; dN = dNint(z)
    dwc = dwcint(z)

    cs = cs2**0.5 ; wc = wc2**0.5
    N = N2**0.5
    kx2 = k_x**2.

    k2 = kx2 + k_z**2.

    tzfirst = (2.*cs*k2*dcs + 2.*wc*dwc)/2.

    tzsecond_up = 2.*(cs2*k2 + wc2)*(2.*tzfirst) - \
            4.*(2.*cs*N2*kx2*dcs + 2.*cs2*N*kx2*dN)

    tzsecond_down = 4.*(((cs2*k2+wc2)**2. - 4.*cs2*N2*kx2)**0.5)

    if mode == 'fast':
        return -(tzfirst + tzsecond_up/tzsecond_down)
    if mode == 'slow':
        return -(tzfirst - tzsecond_up/tzsecond_down)


def dFkx(k_z,z,mode):
    cs2 = cs2int(z) ; N2 = N2int(z)
    wc2 = wc2int(z)

    kx2 = k_x**2.
    k2 = kx2 + k_z**2.

    tkxfirst = 2.*cs2*k_x/2.

    tkxsecond_up = 2.*(cs2*k2+wc2)*2.*cs2*k_x - 8.*cs2*N2*k_x
    tkxsecond_down = 4.*(((cs2*k2+wc2)**2. - 4.*cs2*N2*kx2)**0.5)

    if mode == 'fast':
        return tkxfirst + tkxsecond_up/tkxsecond_down
    if mode == 'slow':
        return tkxfirst - tkxsecond_up/tkxsecond_down


def dFkz(k_z,z,mode):
    cs2 = cs2int(z) ; N2 = N2int(z)
    wc2 = wc2int(z)

    kx2 = k_x**2.
    k2 = kx2 + k_z**2.

    tkzfirst = 2.*cs2*k_z/2.

    tkzsecond_up = 2.*(cs2*k2+wc2)*2.*cs2*k_z
    tkzsecond_down = 4.*(((cs2*k2+wc2)**2. - 4.*cs2*N2*kx2)**0.5)

    if mode == 'fast':
        return tkzfirst + tkzsecond_up/tkzsecond_down
    if mode == 'slow':
        return tkzfirst - tkzsecond_up/tkzsecond_down


# ===========================================

def RungeKutta4th(k_z,x,z,fz,fkx,fkz,ds,mode):
    k1kz = fz(k_z,z,mode)
    k1x = fkx(k_z,z,mode)
    k1z = fkz(k_z,z,mode)

    k2kz = fz(k_z + k1kz*ds*0.5, z + k1z*ds*0.5,mode)
    k2x = fkx(k_z + k1kz*ds*0.5, z + k1z*ds*0.5,mode)
    k2z = fkz(k_z + k1kz*ds*0.5, z + k1z*ds*0.5,mode)

    k3kz = fz(k_z + k2kz*ds*0.5, z + k2z*ds*0.5,mode)
    k3x = fkx(k_z + k2kz*ds*0.5, z + k2z*ds*0.5,mode)
    k3z = fkz(k_z + k2kz*ds*0.5, z + k2z*ds*0.5,mode)

    k4kz = fz(k_z + k3kz*ds, z + k3z*ds,mode)
    k4x = fkx(k_z + k3kz*ds, z + k3z*ds,mode)
    k4z = fkz(k_z + k3kz*ds, z + k3z*ds,mode)

    kznew = k_z + ds/6. * (k1kz + 2.*k2kz + 2.*k3kz + k4kz)
    xnew =  x + ds/6. * (k1x + 2.*k2x + 2.*k3x + k4x)
    znew =  z + ds/6. * (k1z + 2.*k2z + 2.*k3z + k4z)    

    return kznew, xnew, znew



# ==============================================
# ======= CONDICIONES INICIALES ================
# ==============================================


w = 2.5e-3 * (2.*np.pi)




# PARAMETROS PORSIACASO
zc = zint(w**2.)
#k_z2 = ((w**2.-wc2_m)/cs2_m) + (k_x**2./w**2.)*(N2_m - w**2.)
#kz2int = interpolate.interp1d(z_m,k_z2)
#
zinit = np.array([0.5, 1.0, 2.0,5.0,10.0, 15.0])


plt.close('all')
plt.ion()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

colors = plt.cm.viridis(np.linspace(0,1,len(zinit)))
ax1.axis('equal')




kx2_c = (w**2.)*(w**2. - wc2_m)/(cs2_m*(w**2. - N2_m))
H = P_m/(g*rho_m)
Hint = interpolate.interp1d(z_m,H)


ax2.plot(2*H*(kx2_c**0.5),w/(wc2_m**0.5),'k')
ax2.plot(2.*H*(kx2_c**0.5),(N2_m/wc2_m)**0.5,'--k')
ax2.plot(2.*H*(kx2_c**0.5),(cs2_m**0.5)*(kx2_c**0.5)/(wc2_m**0.5),'--k')




c = 0
for z0 in zc - abs(zc)*zinit:

    ds = 1.
    s = np.arange(0.,50000+ds,ds)

    xs = np.ones(s.shape[0])
    xs[0] = 0.0
    zs = np.ones(s.shape[0])
    zs[0] = z0 


    k_x = w/(cs2int(z0)**0.5)
    print(k_x)
    #kx = ((w**2./csint(z0)**2.)*(w**2. - wcint(z0)**2.)/(w**2. - Nint(z0)**2.))**0.5

    kx = np.ones(s.shape[0])*k_x
    kz = np.ones(s.shape[0])
    kz[0] = 0.


    for i in range(s.shape[0]-1):

        # (k_z,x,z,fz,fkx,fkz,ds,mode)
        kz[i+1], xs[i+1], zs[i+1] = \
                RungeKutta4th(kz[i],xs[i],zs[i],dFz,dFkx,dFkz,ds,'fast')



    ax1.plot(xs,zs,label = r'$z_0$ = ' + str(round(z0,2)) + ' km',color = colors[c])

    ax1.set_title('Trayectoria de la onda')
    ax1.set(xlabel = 'x [km]', ylabel = 'z [km]')
    
    ax2.plot(2.*Hint(zs)*k_x,w/(wc2int(zs)**0.5),',',color = colors[c])
    ax2.set(xlabel = r'$2Hk_x$', ylabel = r'$\omega/\omega_c$')

    c += 1


ax1.plot(xs,np.ones(xs.shape[0])*zc,':k')
ax1.legend()


